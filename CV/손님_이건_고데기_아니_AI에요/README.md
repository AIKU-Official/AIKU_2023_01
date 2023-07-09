Member : **구은아**, 강지원, 민재원, 최우석

## 프로젝트 소개

source image의 얼굴에 target image의 머리스타일을 합성하는 모델을 만드는 프로젝트입니다. target image는 어떠한 구도로 틀어져 있어도 source image에 합성될 수 있도록 정면으로 구도를 변환하는 과정을 포함합니다.

1) 3D rotation 모델을 사용하여 target image의 구도를 정면으로 돌립니다.

2) ERSGAN 모델을 사용하여 rotated image의 해상도와 크기를 올립니다.

3) hairstyle_transfer 모델을 사용하여 source image와 변환된 target image를 합성합니다.


## Dataset
