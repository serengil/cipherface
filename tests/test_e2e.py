# built-in dependencies
import tempfile
import os

# 3rd party dependencies
import pytest

# project dependencies
from cipherface import CipherFace, CipherFaceLite
from cipherface.commons.logger import Logger

logger = Logger(__name__)


temp_dir = tempfile.gettempdir()  # system's temp directory
private_key_path = os.path.join(temp_dir, "private.txt")
public_key_path = os.path.join(temp_dir, "public.txt")


def test_phe_e2e():
    model_names = ["VGG-Face", "Facenet", "Facenet512"]
    algorithm_names = ["Paillier", "Damgard-Jurik", "Okamoto-Uchiyama"]
    for model_name in model_names:
        for algorithm_name in algorithm_names:
            onprem = CipherFaceLite(model_name=model_name, algorithm_name=algorithm_name)

            img_paths = ["dataset/img1.jpg", "dataset/img2.jpg", "dataset/img3.jpg"]
            database = {}
            for img_path in img_paths:
                embeddings = onprem.securely_embed(img_path=img_path)
                # return type should be list of str for single input
                assert isinstance(embeddings, list)
                for embedding in embeddings:
                    assert isinstance(embedding, str)
                    database[img_path] = embedding
                    break

            onprem.export_private_key(private_key_path)
            onprem.export_public_key(public_key_path)

            # cloud uses public key
            cloud = CipherFaceLite(
                model_name=model_name,
                algorithm_name=algorithm_name,
                cryptosystem=public_key_path,
            )
            target_path = "dataset/target.jpg"
            target_embedding = cloud.represent(target_path)[0]

            pivot = {}
            for img_path in img_paths:
                encrypted_similarity = cloud.encrypted_compare(database[img_path], target_embedding)

                # confirm that cloud cannot decrypt this
                with pytest.raises(ValueError, match="must have private key"):
                    _ = cloud.restore(encrypted_similarity)

                pivot[img_path] = encrypted_similarity

            # on prem decrypts the similarities
            expected_classifications = [True, True, False]
            for idx, img_path in enumerate(img_paths):
                decrypted_similarity = onprem.restore(pivot[img_path])
                is_verified = onprem.verify(decrypted_similarity)
                logger.debug(
                    f"Similarity between {img_path} and target: {decrypted_similarity} - {is_verified}"
                )
                assert (
                    is_verified is expected_classifications[idx]
                ), f"{img_path} is misclassified. Expected {expected_classifications[idx]}, but got {is_verified}"

            logger.info(f"✅ phe e2e test done for {model_name} - {algorithm_name}")


def test_fhe_e2e():
    model_names = ["VGG-Face", "Facenet", "Facenet512"]
    distance_metrics = ["euclidean", "cosine"]

    for model_name in model_names:
        for distance_metric in distance_metrics:
            # on prem generates cryptosystem with private - public key pair
            onprem = CipherFace(model_name=model_name, distance_metric=distance_metric)

            img_paths = ["dataset/img1.jpg", "dataset/img2.jpg", "dataset/img3.jpg"]

            database = {}
            for img_path in img_paths:
                embeddings = onprem.securely_embed(img_path=img_path)
                # return type should be list of str for single input
                assert isinstance(embeddings, list)
                for embedding in embeddings:
                    assert isinstance(embedding, str)
                    database[img_path] = embedding
                    break

            onprem.export_private_key(private_key_path)
            onprem.export_public_key(public_key_path)

            # cloud uses public key to securely embed the image
            cloud = CipherFace(
                model_name=model_name,
                distance_metric=distance_metric,
                cryptosystem=public_key_path,
            )

            target_path = "dataset/target.jpg"
            target_embedding = cloud.securely_embed(img_path=target_path)[0]

            # even though cloud encrypted target embedding, it cannot decrypt it
            with pytest.raises(ValueError, match="doesn't hold a secret_key"):
                _ = cloud.restore(target_embedding)

            pivot = {}
            for img_path in img_paths:
                encrypted_distance = cloud.encrypted_compare(database[img_path], target_embedding)
                # confirm that cloud cannot decrypt this
                with pytest.raises(ValueError, match="doesn't hold a secret_key"):
                    _ = cloud.restore(encrypted_distance)

                pivot[img_path] = encrypted_distance

            # on prem decrypts the distances
            expected_classifications = [True, True, False]
            for idx, img_path in enumerate(img_paths):
                decrypted_distance = onprem.restore(pivot[img_path])
                is_verified = onprem.verify(decrypted_distance)
                logger.debug(
                    f"Distance between {img_path} and target: {decrypted_distance} - {is_verified}"
                )
                assert is_verified is expected_classifications[idx], (
                    f"{img_path} is misclassified. Expected {expected_classifications[idx]},"
                    f"but got {is_verified}"
                )

            logger.info(f"✅ e2e euclidean test done for {model_name} - {distance_metric}")


def test_many_inputs_to_securely_embed():
    img_paths = ["dataset/img1.jpg", "dataset/img2.jpg", "dataset/img3.jpg"]

    onprem = CipherFace(model_name="Facenet", distance_metric="euclidean")
    embeddings = onprem.securely_embed(img_path=img_paths)
    # return type should be list of list of str for multiple inputs
    assert isinstance(embeddings, list)
    for embedding in embeddings:
        assert isinstance(embedding, list)
        for emb in embedding:
            assert isinstance(emb, str)

    logger.info("✅ securely embed multiple inputs test done")
