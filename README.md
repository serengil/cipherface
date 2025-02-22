# CipherFace

CipherFace is a fully homomorphic encryption-driven python framework for secure cloud-based facial recognition. It combines [DeepFace](https://github.com/serengil/deepface) and [TenSEAL](https://github.com/OpenMined/TenSEAL) libraries.

## On Prem Encryption

When you initialize a CipherFace object, it sets up an FHE cryptosystem. Currently, CipherFace supports the [`VGG-Face`](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/), [`Facenet`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/), and [`Facenet512`](https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/) facial recognition models, as well as Euclidean and cosine distance metrics.

```python
from cipherface import CipherFace

# build a cryptosystem
onprem = CipherFace(
    facial_recognition_model="Facenet",
    distance_metric="euclidean",
)

# export keys of built cryptosystem
onprem.export_private_key("private.txt")
onprem.export_public_key("public.txt")

# create vector embedding for 1st image and encrypt in one shot
source_embedding_encrypted = onprem.securely_embed(img_path="dataset/img1.jpg")
```

The on-prem system should generate embeddings for its facial database and encrypt them in advance. This process only needs to be done once to extract the encrypted embeddings. Once encrypted, these embeddings can be securely stored in the cloud.

## Encrypted Distance Calculation On Cloud

The cloud can also generate vector embeddings and encrypt them since encryption only requires a public key. Additionally, it can compute the encrypted distance between a recently generated encrypted embedding and an encrypted embedding created on the on-prem side.

```python
# cloud loads cryptosystem with public key
onprem = CipherFace(
    facial_recognition_model="Facenet",
    distance_metric="euclidean",
    cryptosystem="public.txt",
)

# create vector embedding for target image and encrypt in one shot
target_embedding_encrypted = cloud.securely_embed(img_path="dataset/target.jpg")[0]

encrypted_distance = cloud.calculate_encrypted_distance(
    target_embedding_encrypted,
    source_embedding_encrypted
)
```

## On Prem Verification

Once the cloud calculates the encrypted distance, only the on-prem system can decrypt it since it holds the private key of the cryptosystem. This allows the on-prem system to determine whether the source and target images belong to the same person or different individuals.

```python
# on prem loads cryptosystem with private key
onprem = CipherFace(
    facial_recognition_model="Facenet",
    distance_metric="euclidean",
    cryptosystem="private.txt",
)

# on prem restores distance
decrypted_distance = onprem.restore(encrypted_distance)

# verification
is_verified = onprem.verify(decrypted_distance)

if is_verified is True:
    print("they are same person")
else:
    print("they are different persons")
```

In this setup, the cloud system performs the distance calculation, utilizing most of the computational power. The on-prem system, holding the private key, is only responsible for decrypting the distances to determine whether the images belong to the same person or different individuals.

## Contribution

Pull requests are more than welcome! If you are planning to contribute a large patch, please create an issue first to get any upfront questions or design decisions out of the way first.

Before creating a PR, you should run the unit tests and linting locally by running `make test && make lint` command. Once a PR sent, GitHub test workflow will be run automatically and unit test and linting jobs will be available in [GitHub actions](https://github.com/serengil/cipherface/actions) before approval.

## Support

There are many ways to support a project - starring⭐️ the GitHub repo is just one 🙏

If you do like this work, then you can support it financially on [Patreon](https://www.patreon.com/serengil?repo=deepface), [GitHub Sponsors](https://github.com/sponsors/serengil) or [Buy Me a Coffee](https://buymeacoffee.com/serengil).

<a href="https://www.patreon.com/serengil?repo=deepface">
<img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/patreon.png" width="30%" height="30%">
</a>

<a href="https://github.com/sponsors/serengil">
<img src="https://raw.githubusercontent.com/serengil/deepface/refs/heads/master/icon/github_sponsor_button.png" width="37%" height="37%">
</a>

<a href="https://buymeacoffee.com/serengil">
<img src="https://raw.githubusercontent.com/serengil/deepface/master/icon/bmc-button.png" width="25%" height="25%">
</a>

## Licence

CipherFace is licensed under the MIT License - see [`LICENSE`](https://github.com/serengil/cipherface/blob/master/LICENSE) for more details.