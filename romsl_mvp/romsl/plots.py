from html import escape

import numpy as np


def _panel(elements, title, x, y, traces, edges=(), *, scatter=False):
    width, height = 265, 145
    elements.extend([
        f'<text x="{x}" y="{y - 6}" font-size="12">{escape(title)}</text>',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="white" stroke="#cbd5e1"/>',
    ])
    finite = np.concatenate(traces)
    finite = finite[np.isfinite(finite).all(axis=1)]
    if not len(finite):
        return
    low, high = finite.min(axis=0), finite.max(axis=0)
    scale = np.maximum(high - low, 1e-9)
    if scatter:
        scale[:] = scale.max()
    palette = ('#2563eb', '#dc2626', '#059669', '#a16207')
    for trace, color in zip(traces, palette):
        normalized = (trace - low) / scale
        px = x + 7 + normalized[:, 0] * (width - 14)
        py = y + height - 7 - normalized[:, 1] * (height - 14)
        valid = np.isfinite(px) & np.isfinite(py)
        if scatter:
            for start, end in edges:
                if valid[start] and valid[end]:
                    elements.append(f'<line x1="{px[start]:.1f}" y1="{py[start]:.1f}" '
                                    f'x2="{px[end]:.1f}" y2="{py[end]:.1f}" stroke="#475569"/>')
            for i in np.flatnonzero(valid):
                elements.append(f'<circle cx="{px[i]:.1f}" cy="{py[i]:.1f}" r="2.6" fill="{color}"/>')
                if edges:
                    elements.append(f'<text x="{px[i]+3:.1f}" y="{py[i]-3:.1f}" font-size="9">{i}</text>')
        else:
            starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
            ends = np.flatnonzero(valid & ~np.r_[valid[1:], False]) + 1
            for start, end in zip(starts, ends):
                coords = ' '.join(f'{a:.1f},{b:.1f}' for a, b in zip(px[start:end], py[start:end]))
                elements.append(f'<polyline points="{coords}" fill="none" stroke="{color}"/>')


def write_recording_plot(path, recording, joints, fps):
    pose = recording.pose
    tracks = recording.points_per_frame
    hips = [joints['left_hip'], joints['right_hip']]
    root = pose[:, hips].mean(axis=1)
    seconds = np.arange(len(pose)) / fps
    groups = list(joints['groups'].values())
    edges = list(joints['torso_edges'])
    for group in groups:
        edges.extend(zip(group[:-1], group[1:]))
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="890" height="590" viewBox="0 0 890 590" '
        'font-family="sans-serif">',
        '<rect width="890" height="590" fill="#f8fafc"/>',
        f'<text x="20" y="24" font-size="16">{escape(recording.recording_id)} | independent axes, uncalibrated</text>',
    ]
    for row, frame in enumerate((len(pose) // 3, 2 * len(pose) // 3)):
        top = 54 + row * 174
        _panel(elements, f'Frame {recording.frame_ids[frame]} - radar x/z', 20, top,
               [tracks[frame][:, [0, 2]]], scatter=True)
        _panel(elements, f'Frame {recording.frame_ids[frame]} - candidate skeleton x/y', 307, top,
               [pose[frame][:, [0, 1]]], edges, scatter=True)
        _panel(elements, f'Frame {recording.frame_ids[frame]} - candidate skeleton x/z', 594, top,
               [pose[frame][:, [0, 2]]], edges, scatter=True)
    _panel(elements, 'Point count over time', 20, 402, [np.c_[seconds, [len(frame) for frame in tracks]]])
    _panel(elements, 'Candidate root x (blue), y, z', 307, 402, [np.c_[seconds, axis] for axis in root.T])
    limb_traces = []
    for group in groups:
        relative = pose[:, group] - root[:, None, :]
        speed = np.linalg.norm(np.diff(relative, axis=0), axis=-1).mean(axis=1) * fps
        limb_traces.append(np.c_[seconds[1:], speed])
    _panel(elements, 'Candidate limb speed L/R arm, L/R leg', 594, 402, limb_traces)
    elements.extend([
        '<text x="20" y="570" font-size="12">Time panels run from start to end of the recording. '
        'Geometry and candidate joint indices still require review.</text>', '</svg>',
    ])
    path.write_text('\n'.join(elements) + '\n', encoding='utf-8')
