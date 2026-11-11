"""Bounded online event aggregation; EOF is explicit and idempotent."""
from collections import deque
import torch


class FrameBuffer:
    def __init__(self, capacity, feature_dim):
        self.capacity, self.feature_dim = capacity, feature_dim
        self.frames = deque()
        self.prefix_sum, self.prefix_count = None, 0
        self.next_index = 0
        self.start = 0

    @property
    def stored_frames(self):
        return len(self.frames)

    @property
    def earliest_retroactive(self):
        return self.frames[0][0] if self.frames else self.next_index

    def append(self, index, h):
        if index != self.next_index or h.shape != (self.feature_dim,):
            raise ValueError('buffer requires contiguous aligned frames')
        self.next_index += 1
        if self.prefix_sum is None:
            self.prefix_sum = torch.zeros_like(h)
        self.frames.append((index, h.detach().clone()))
        if len(self.frames)>self.capacity:
            _, old = self.frames.popleft()
            self.prefix_sum += old
            self.prefix_count += 1

    def summary(self):
        if self.prefix_sum is None:
            raise ValueError('empty buffer')
        total, count = self.prefix_sum.clone(), self.prefix_count
        for _, h in self.frames:
            total += h
            count += 1
        return total, count

    def split(self, boundary):
        if not self.start < boundary <= self.next_index or boundary < self.earliest_retroactive:
            raise ValueError('boundary outside reconstructable nonempty event')
        total, count = self.prefix_sum.clone(), self.prefix_count
        while self.frames and self.frames[0][0] < boundary:
            _, h = self.frames.popleft()
            total += h
            count += 1
        self.prefix_sum.zero_(); self.prefix_count = 0; self.start = boundary
        return total, count


class EventStream:
    def __init__(self, model, recording_id='recording', minimum_segment_frames=1):
        if model.training:
            raise ValueError('call model.eval() before streaming')
        if type(minimum_segment_frames) is not int or not 1 <= minimum_segment_frames <= model.config.buffer_frames:
            raise ValueError('baseline duration must fit the retrospective buffer')
        self.model, self.recording_id = model, str(recording_id)
        self.buffer = FrameBuffer(model.config.buffer_frames, model.config.feature_dim)
        self.state, self.index, self.event_index = None, 0, 0
        self.closed, self.low_point = False, 0
        self.mass = 0.
        self.weighted = None
        self.pending = None
        self.minimum_segment_frames = minimum_segment_frames
        self.current_class, self.candidate_class, self.candidate_count = None, None, 0

    @property
    def stored_frames(self):
        return self.buffer.stored_frames

    def _event(self, start, stop, total, count, token, confirmed, closure, quality='not_requested'):
        if count != stop-start or count <= 0:
            raise RuntimeError('event frame ownership is inconsistent')
        probability = self.model.classifier(token).softmax(-1)
        period = self.model.config.frame_period_s
        row = dict(recording_id=self.recording_id, event_id=self.event_index,
                   start_frame=start, stop_frame=stop, start_s=start*period, stop_s=stop*period,
                   duration_s=count*period, confirmed_at_frame=confirmed, confirmed_at_s=confirmed*period,
                   class_id=int(probability.argmax()), confidence=float(probability.max()),
                   probabilities=probability.cpu().tolist(), feature_mean=(total/count).cpu().tolist(),
                   closure=closure, boundary_quality=quality, time_basis='nominal')
        self.event_index += 1
        return row

    @staticmethod
    def _public(row):
        return {k:v for k,v in row.items() if not k.startswith('_')}

    def _baseline_event(self, start, stop, total, count, confirmed, closure):
        event = self._event(start, stop, total, count, total/count, confirmed, closure)
        event['class_id'] = self.current_class
        event['confidence'] = event['probabilities'][self.current_class]
        return event

    @torch.no_grad()
    def push(self, features):
        if self.closed:
            raise ValueError('cannot push after EOF')
        if len(features) == 0:
            return []
        features = self.model.validate_features(features)
        emitted = []
        for h in features:
            i = self.index
            self.buffer.append(i, h)
            if self.model.mode == 'frame-baseline':
                predicted = int(self.model.classifier(h).argmax())
                if self.current_class is None:
                    self.current_class = predicted
                if predicted == self.current_class:
                    self.candidate_class, self.candidate_count = None, 0
                else:
                    self.candidate_count = self.candidate_count+1 if predicted == self.candidate_class else 1
                    self.candidate_class = predicted
                    if self.candidate_count >= self.minimum_segment_frames:
                        boundary = i-self.candidate_count+1
                        start = self.buffer.start
                        total, count = self.buffer.split(boundary)
                        emitted.append(self._baseline_event(start, boundary, total, count, i+1, 'fire'))
                        self.current_class, self.candidate_class, self.candidate_count = predicted, None, 0
            elif self.model.mode == 'lif':
                if self.state is None or float(self.state.dynamics.voltage) <= .1*self.model.config.theta:
                    self.low_point = i
                self.state, info = self.model.lif_step(h, self.state)
                if bool(info['hard']):
                    boundary, quality = i, 'firing_time'
                    if self.model.config.retrospective:
                        if self.buffer.start < self.low_point <= i and self.low_point >= self.buffer.earliest_retroactive:
                            boundary, quality = self.low_point, 'retrospective'
                        else:
                            quality = 'fallback_firing_time'
                    if boundary > self.buffer.start:
                        start = self.buffer.start
                        total, count = self.buffer.split(boundary)
                        emitted.append(self._event(start, boundary, total, count, total/count, i+1, 'fire', quality))
                        total, count = self.buffer.summary()
                        self.state.content_sum, self.state.content_count = total, total.new_tensor(float(count))
                        self.state.reference, self.state.recent = total/count, total/count
                        self.low_point = boundary
            else:
                alpha, self.state = self.model.cif_step(h, self.state)
                weight = float(alpha)
                if self.weighted is None:
                    self.weighted = torch.zeros_like(h)
                take = min(weight, 1-self.mass)
                self.weighted += take*h
                self.mass += take
                if self.mass >= 1-1e-6:
                    if self.pending is not None:
                        self.pending['confirmed_at_frame'] = i+1
                        self.pending['confirmed_at_s'] = (i+1)*self.model.config.frame_period_s
                        emitted.append(self._public(self.pending))
                    start = self.buffer.start
                    total, count = self.buffer.split(i+1)
                    token = self.weighted/self.mass
                    row = self._event(start, i+1, total, count, token, i+1, 'fire')
                    row.update(_total=total, _count=count, _token_sum=self.weighted.clone(), _mass=self.mass)
                    self.pending = row
                    remainder = weight-take
                    self.weighted, self.mass = remainder*h, remainder
            self.index += 1
        return emitted

    @torch.no_grad()
    def finish(self):
        if self.closed:
            return []
        self.closed = True
        if not self.index:
            return []
        total, count = self.buffer.summary()
        if self.model.mode == 'frame-baseline':
            return [self._baseline_event(self.buffer.start,self.index,total,count,self.index,'eof')]
        if self.model.mode == 'lif':
            return [self._event(self.buffer.start, self.index, total, count, total/count, self.index, 'eof')]
        if self.pending is None:
            token = self.weighted/max(self.mass, 1e-12)
            return [self._event(0, self.index, total, count, token, self.index, 'eof_low_evidence')]
        if count and self.mass >= self.model.config.tail_threshold:
            self.pending['confirmed_at_frame'] = self.index
            self.pending['confirmed_at_s'] = self.index*self.model.config.frame_period_s
            tail = self._event(self.buffer.start, self.index, total, count, self.weighted/self.mass, self.index, 'eof')
            return [self._public(self.pending), tail]
        old = self.pending
        token = (old['_token_sum']+self.weighted)/(old['_mass']+self.mass)
        # Reuse the existing ID when the last event absorbs a low-mass tail.
        self.event_index = old['event_id']
        merged = self._event(old['start_frame'], self.index, old['_total']+total, old['_count']+count,
                             token, self.index, 'eof_tail_merged')
        return [merged]
