# Radar + GRU two zero-update diagnostics

> Execute inline using executing-plans and systematic-debugging, with TDD and one final fresh read-only review. User explicitly authorizes only the first two diagnostics from docs/radar+gru融合方案.md.

**Goal:** Export an onset-review set and measure actual causal dual-source candidate-pool recall versus density, without training or changing labels/models/triggers.

**Architecture:** Read immutable existing500__radar/recurrent__state predictions. Keep Radar at its existing fit8 threshold. Sweep GRU threshold0:.01:1 on fit, form causal cross-source episodes, choose minimum-density fit points reaching80/85/90/95% recall. Export150 balanceddev rawcanonicalMD/RD clips and a review template. No residualB, no Verify probe or controller training.

**Tech stack:** Existing interview Python/pytest locally, seer Python onserver; NumPy/Matplotlib, original fixed peak trigger/matcher.

## Rulings and interfaces

- B is a descriptive zero-update diagnostic; serious onset misalignment cannot be decided before humanreview. Complete B on unchanged protocolGT, with no progression to Verify until onsetaudit resolved. No physical truth relabeling.
- Radar threshold is existing500__radar priorfit8; allotherpeakrules remainlag1/.18,NMS.36,neighbor-high. GRU is priorindependentrecurrent__state, neverresidualB. Commonvalidcontext and one-to-one ±.72s matcher unchanged.
- Online merge: first-arrival episode time fixed tofirstemission; a later opposite-source peak within.36s attaches to one unmatched opposite-source episode, nearestprioranchor thenearliest tie. Atmostonepeak per source per episode. Same-source peaks stayindependent. Noaveraging/backdating, no transitive slidingwindows, no retroactive sourceevidence. Simultaneous tieRadarfirst. Attachmetadata arrivalstimes retained.
- Fit selection usesonly18fit streams. For eachtarget80/85/90/95%,chooseminimumdensity amongpoints reachingtarget, tiehigherrecallthenhigherthreshold. 85%primary; ifunattainableexplicitmissing, reportmaxfitrecall. Savefitlock beforeone dev readout atuniquelyselected thresholds, pluszero endpoint reference clearlyseparate. Curveissfitonly; devnotusedtoselect.
- A dev50 Radarfit8 correctlymatchedGT,50 GRU-only protocolfalse zero-threshold peaks fartherthan.36s fromanyRadarzeropeak,50ordinary regions withnoGT±2s,noRadar/GRUfit8 emission±2s,andconstantlabel overclip. Seed42balancedroundrobinacross8cells; auditnotrepresentativepopulationestimate. Exactsampledefinition andeligiblecounts saved; shortages explicit, no replacement/fakecases.
- A clipscenter±2s; exportfullcanonicalMD/RD/labels/timestamps andscores; inspectonlyselected8devrawpaths andtheirpaired source recording folders for synchronousvideomodalities. Noofficialholdoutarrayreads. Videoavailabilityrequiresactualsynchronizationevidence; filenamesalone insufficient. Withoutindependentreference:physicalonset/offset/confidencenotfabricated, pendingreview; proxyonly.
- A command_GT forfalse/stable nearestGT iscontext only, notassertionthatpeakisatransition. Template permitsno-observabletransitionandmultipleproxyonsets; offsetdistributionsonlyfromreviewed independentlysupportedonset, notfromscorepeaks.
- Separatecode/evidence under .audit-staging; noGitcheckout, noeditingoriginalweights/caches/productioncode. SHA256 verification source andarchive; keeprawartifacts. SourcecodeguardA/Bneverconstructsmodel/optimizer.

## Tasks

- [ ] Task1 RED→GREEN causalmerge,fit-onlymin-densityselection,samplingdisjointness/shortage cases; writecore/sweep, verifyagainstoriginalemitter/matcher andsourceC.
- [ ] Task2 selectbalanced150devcases; pinnedrawdataclip export andmodalityinventory onserver; plots/contactsheet/reviewtemplate, nofakephysicalonsetstats.
- [ ] Task3 freshread-onlyreview, archiveSHA/memberverification, independentpoolreconstruction andfullreport/fitcurve; nofurthertraining.
