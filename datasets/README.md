# Dataset protocol

TAPF-MIN does not redistribute biometric datasets from this repository.

## Primary benchmark: CREMA-D

Purpose: same-video evaluation of actor identity leakage and facial-emotion utility.

Required protocol:
- retain actor ID from the filename/metadata;
- use subject-disjoint train/validation/test partitions;
- never allow one actor to occur in multiple splits;
- evaluate the exact same clips under Original, Blur, Pixelation, Static TAPF, Adaptive TAPF and TAPF-MIN;
- report per-method identity leakage and emotion utility.

Official project repository: CheyneyComputerScience/CREMA-D on GitHub. Review its license and data terms before use.

## External privacy benchmark: LFW

Purpose: independent face-verification/privacy evaluation. Use an established LFW protocol or clearly document any custom subject-disjoint protocol. Do not tune privacy thresholds on the final test identities.

## External utility benchmark: FER+ / FER2013

Purpose: independent facial-expression utility evaluation. FER+ supplies improved annotations but not the original FER2013 images; obtain the image data through an authorized source and respect its terms.

## Publication rule

A dataset being publicly downloadable does not mean its biometric samples should be copied into this repository. Store only scripts, manifests, hashes/splits where permitted, and derived aggregate metrics that do not expose participant biometric media.
