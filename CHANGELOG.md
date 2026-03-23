# Changelog

## [0.1.3](https://github.com/spectrseq/spectrseqtools/compare/v0.1.2...v0.1.3) (2026-02-15)


### Bug Fixes

* catch exceptions during lp based filtering ([#52](https://github.com/spectrseq/spectrseqtools/issues/52)) ([254e25a](https://github.com/spectrseq/spectrseqtools/commit/254e25a56e24a86a3b0d7e3275413a2a844d4f1b))

## [0.1.2](https://github.com/spectrseq/spectrseqtools/compare/v0.1.1...v0.1.2) (2026-01-30)


### Bug Fixes

* remove superfluous cython dependency ([#50](https://github.com/spectrseq/spectrseqtools/issues/50)) ([5b67a9d](https://github.com/spectrseq/spectrseqtools/commit/5b67a9d7811ef85d4b6de2d2f21a1cdb1ce3e9d0))


### Documentation

* readme ([468eac3](https://github.com/spectrseq/spectrseqtools/commit/468eac37bcd44f0ce3d6aa421ac92ca28b5e564f))

## [0.1.1](https://github.com/spectrseq/spectrseqtools/compare/v0.1.0...v0.1.1) (2026-01-30)


### Bug Fixes

* add package description ([472b3a9](https://github.com/spectrseq/spectrseqtools/commit/472b3a9fd6d3aef9b04e9222f459c9fdbd463faf))

## 0.1.0 (2026-01-30)


### Features

* add modification rate ([#12](https://github.com/spectrseq/spectrseqtools/issues/12)) ([450e586](https://github.com/spectrseq/spectrseqtools/commit/450e586c6a764e3098300b2ffc95b3146d8e5d26))
* add option to set lp time limits ([#40](https://github.com/spectrseq/spectrseqtools/issues/40)) ([555eef7](https://github.com/spectrseq/spectrseqtools/commit/555eef7703a4af3b30f5e96ca8124dadb153bdd3))
* add option to set output directory ([#44](https://github.com/spectrseq/spectrseqtools/issues/44)) ([0119188](https://github.com/spectrseq/spectrseqtools/commit/0119188753e54c1e6114b3e0b8adecf2353ce850))
* add preprocessing ([#21](https://github.com/spectrseq/spectrseqtools/issues/21)) ([40c5214](https://github.com/spectrseq/spectrseqtools/commit/40c521476289f6e6a017c5f3ba81a55c0faf0fbf))
* additionally output sequence with all alternate nucleotides ([#42](https://github.com/spectrseq/spectrseqtools/issues/42)) ([675a310](https://github.com/spectrseq/spectrseqtools/commit/675a31004fd52d751640fdeaa43b4a533c8d13d6))
* allow both raw and preprocessed data as input ([#23](https://github.com/spectrseq/spectrseqtools/issues/23)) ([d4d353a](https://github.com/spectrseq/spectrseqtools/commit/d4d353a97f1d95f0b171e9cbd4e2a5b5096ec7ba))
* allow singleton selection for tsv input ([#29](https://github.com/spectrseq/spectrseqtools/issues/29)) ([0beb9cd](https://github.com/spectrseq/spectrseqtools/commit/0beb9cd7bd740d67f98cea38011ac62cc2df2e4c))
* allow using lp based filter on terminal fragments ([#45](https://github.com/spectrseq/spectrseqtools/issues/45)) ([fac53a6](https://github.com/spectrseq/spectrseqtools/commit/fac53a6bc331617b5671205a7bbb21e5635fb8b0))
* build skeleton and use that to constrain the MILP ([15b21bd](https://github.com/spectrseq/spectrseqtools/commit/15b21bd1f7b439d7f48ff0c41bdf8c660c061177))
* enable recomputing of dp table ([#22](https://github.com/spectrseq/spectrseqtools/issues/22)) ([5f7ffd7](https://github.com/spectrseq/spectrseqtools/commit/5f7ffd75edeefa13fd4c52f34b368af117bdc20c))
* establish standard unit masses for simplified nucleotide consideration ([#15](https://github.com/spectrseq/spectrseqtools/issues/15)) ([ab5206d](https://github.com/spectrseq/spectrseqtools/commit/ab5206d5a0ade3e9db03dd86338cd7e4ca5c4d94))
* estimate sequence length ([#20](https://github.com/spectrseq/spectrseqtools/issues/20)) ([a6bc2d6](https://github.com/spectrseq/spectrseqtools/commit/a6bc2d60dd01d578e5c9107d7ff5f678865d3657))
* explain observed masses or mass differences using dynamic programming ([#3](https://github.com/spectrseq/spectrseqtools/issues/3)) ([c36b8cc](https://github.com/spectrseq/spectrseqtools/commit/c36b8cc474194e5f6f9ec129318af14507de10b9))
* extend dp to nucleotides ([#9](https://github.com/spectrseq/spectrseqtools/issues/9)) ([0b5f411](https://github.com/spectrseq/spectrseqtools/commit/0b5f411edac2f6e88ec46700fb438638623a0e82))
* filter early retention times with noise peaks ([#32](https://github.com/spectrseq/spectrseqtools/issues/32)) ([b7b1301](https://github.com/spectrseq/spectrseqtools/commit/b7b130113844397d42eeb14c09abdec303e09712))
* filter fragments with lp ([#10](https://github.com/spectrseq/spectrseqtools/issues/10)) ([0b947fc](https://github.com/spectrseq/spectrseqtools/commit/0b947fc04a267bf7ed8ab76fd27ff934f9bd49a3))
* filter fragments with sequence mass ([#16](https://github.com/spectrseq/spectrseqtools/issues/16)) ([d51e573](https://github.com/spectrseq/spectrseqtools/commit/d51e5733b2963a1cce827527c3fc1f5bb1f12626))
* group same-weight fragments during skeleton building ([#17](https://github.com/spectrseq/spectrseqtools/issues/17)) ([e3d34f5](https://github.com/spectrseq/spectrseqtools/commit/e3d34f57b3361de11529339c5fa1cbf5a9b33ece))
* improve sequence length estimation with lp based selection ([#26](https://github.com/spectrseq/spectrseqtools/issues/26)) ([b102ffb](https://github.com/spectrseq/spectrseqtools/commit/b102ffbd844a73191cd379d111aa085ce501c544))
* plot different fragment classes individually ([#46](https://github.com/spectrseq/spectrseqtools/issues/46)) ([281a848](https://github.com/spectrseq/spectrseqtools/commit/281a84802ad6d8d0c11f745cdafad7445676afd5))
* reduce alphabet after skeleton building ([#19](https://github.com/spectrseq/spectrseqtools/issues/19)) ([4e95659](https://github.com/spectrseq/spectrseqtools/commit/4e95659685b1e07b5bf3cb94c2d285ff867c2e9c))
* set intensity cutoff based on percentile ([#41](https://github.com/spectrseq/spectrseqtools/issues/41)) ([cfc1085](https://github.com/spectrseq/spectrseqtools/commit/cfc1085f096f55439e2b6999fe7f8ea28ba22261))


### Bug Fixes

* allow using cli ([#13](https://github.com/spectrseq/spectrseqtools/issues/13)) ([09a65af](https://github.com/spectrseq/spectrseqtools/commit/09a65afa989d3912715505a209e35ed6cab7154a))
* catch exceptions during lp initialization for ambiguity removal ([#39](https://github.com/spectrseq/spectrseqtools/issues/39)) ([4bf108b](https://github.com/spectrseq/spectrseqtools/commit/4bf108b2e415d1c1edfd8e95061d21651555ff28))
* catch exceptions if no prediction is possible ([#30](https://github.com/spectrseq/spectrseqtools/issues/30)) ([02405e1](https://github.com/spectrseq/spectrseqtools/commit/02405e16f53a72a5424232f1d8f18d001934c05d))
* catch exceptions while solving lp ([#38](https://github.com/spectrseq/spectrseqtools/issues/38)) ([7ea9737](https://github.com/spectrseq/spectrseqtools/commit/7ea97376bb708adf9abd93864191057a1b0d08da))
* catch nonetype bases during lp evaluation ([#37](https://github.com/spectrseq/spectrseqtools/issues/37)) ([4c5c695](https://github.com/spectrseq/spectrseqtools/commit/4c5c695fc16af46b99765e13fa3d69adb0ed9371))
* ensure correct fragment end indices for lp ([#25](https://github.com/spectrseq/spectrseqtools/issues/25)) ([051e744](https://github.com/spectrseq/spectrseqtools/commit/051e7448d0bb32747d83ec5b7966df701c7cfed5))
* ensure usage of correct singleton path ([#35](https://github.com/spectrseq/spectrseqtools/issues/35)) ([51264eb](https://github.com/spectrseq/spectrseqtools/commit/51264eb74d65533955781064bcbc99309a408441))
* filter singletons by combined tag mass ([#24](https://github.com/spectrseq/spectrseqtools/issues/24)) ([7b6ab9a](https://github.com/spectrseq/spectrseqtools/commit/7b6ab9a04282f706eebaa56ca5b625e313efd2e5))
* map singletons to mass representative to avoid filtering them out ([#33](https://github.com/spectrseq/spectrseqtools/issues/33)) ([82de7e8](https://github.com/spectrseq/spectrseqtools/commit/82de7e88b4971c6f2d198d5ae12c6a4b20427bc8))
* plot only one base at each position of global sequence ([#34](https://github.com/spectrseq/spectrseqtools/issues/34)) ([f6ec9a2](https://github.com/spectrseq/spectrseqtools/commit/f6ec9a2880b3fa35cccacbdb6aa5189455c58c11))
* skip fragments with non-explainable masses during ladder building; allow start fragments to reach the end, and vice versa; non certain start or end fragments may also touch start and end and even span the entire sequence ([#7](https://github.com/spectrseq/spectrseqtools/issues/7)) ([9a1f03b](https://github.com/spectrseq/spectrseqtools/commit/9a1f03b5c6eb243399c495ac3463d201dfd04a7b))


### Performance Improvements

* further limit MILP by terminal fragment ladder skeleton ([6e9b7aa](https://github.com/spectrseq/spectrseqtools/commit/6e9b7aa97f3a094b4cbecbb2f2721c2abd444893))
