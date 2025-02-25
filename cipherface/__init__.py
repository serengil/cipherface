# built-in dependencies
import os
from typing import Optional, List, Union
import base64

# 3rd party dependencies
from deepface import DeepFace
import tenseal as ts
import numpy as np
from tqdm import tqdm

# project dependencies
from cipherface.commons.logger import Logger
from cipherface.commons import file_utils


__version__ = "0.0.1"


# pylint: disable=unknown-option-value
class CipherFace:
    def __init__(
        self,
        model_name: str = "VGG-Face",
        face_detector: str = "opencv",
        distance_metric: str = "euclidean",
        cryptosystem: Optional[str] = None,
        security_level: int = 128,
        mode: str = "defensive",
        n: Optional[int] = None,
        q: Optional[List[int]] = None,
        g: Optional[int] = None,
    ):
        """
        Build the CipherFace
        Args:
            model_name (str): facial recognition model name.
                Options: VGG-Face, Facenet, Facenet512
            face_detector (str): The path to the face detector.
                Options: opencv, mtcnn, ssd, dlib, retinaface, mediapipe,
                yolov8, yolov11n, yolov11s, yolov11m, yunet, fastmtcnn or centerface
            distance_metric (str): The distance metric to use for comparing embeddings.
                Options: euclidean, cosine
            cryptosystem (str): The path to the cryptosystem. Generates a random
                private-public key pair if None.
            security_level (int): The security level of the cryptosystem. Default is 128.
            mode (str): The mode of the cryptosystem. Options are offensive and defensive.
                Default is defensive. In HE, you can offer same security level with different
                p, q, g values. Offensive mode is offering same security level but slower.
            n (int): You can set the security level with n, q, g values. P is the polynomial
                modulus degree.
            q (list of int): You can set the security level with n, q, g values. Q is the
                coefficient modulus bit sizes.
            g (int): You can set the security level with n, q, g values. G is the scale.
        """

        self.logger = Logger(__name__)

        self.model_name = model_name

        assert self.model_name in [
            "VGG-Face",
            "Facenet",
            "Facenet512",
        ], f"Unsupported model {self.model_name}"

        self.face_detector = face_detector
        assert self.face_detector in [
            "opencv",
            "mtcnn",
            "ssd",
            "dlib",
            "retinaface",
            "mediapipe",
            "yolov8",
            "yolov11n",
            "yolov11s",
            "yolov11m",
            "yunet",
            "fastmtcnn",
            "centerface",
        ], f"Unsupported face detector {self.face_detector}"

        self.distance_metric = distance_metric
        assert self.distance_metric in [
            "euclidean",
            "cosine",
        ], f"Unsupported distance metric {self.distance_metric}"

        if cryptosystem is None:
            if n is not None and q is not None and g is not None:
                assert len(q) >= 3
                assert q[0] == q[-1]
                assert q[1] == g
            elif security_level == 128 and mode == "defensive":
                n = 2**13
                q = [60, 40, 40, 60]
                g = 2**40
            elif security_level == 128 and mode == "offensive":
                n = 2**14
                q = [31, 60, 60, 60, 60, 60, 60, 31]
                g = 2**60
            elif security_level == 192 and mode == "defensive":
                n = 2**13
                q = [60, 40, 60]
                g = 2**40
            elif security_level == 192 and mode == "offensive":
                n = 2**14
                q = [60, 60, 60, 60, 60]
                g = 2**60
            elif security_level == 256 and mode == "defensive":
                n = 2**13
                q = [30, 30, 30, 30]
                g = 2**30
            elif security_level == 256 and mode == "offensive":
                n = 2**14
                q = [45, 45, 45, 45, 45]
                g = 2**45
            else:
                raise ValueError(f"Security level {security_level} & mode {mode} not supported.")

            context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=n, coeff_mod_bit_sizes=q)
            context.generate_galois_keys()
            context.global_scale = g

            self.context = context
            self.logger.debug("Cryptosystem is just generated.")
        else:
            if os.path.exists(cryptosystem) is False:
                raise FileNotFoundError(f"Cryptosystem not found at {cryptosystem}")

            data = file_utils.read_data(cryptosystem)
            self.context = ts.context_from(data)
            self.logger.debug(f"Cryptosystem is just loaded from {cryptosystem}")

    def export_public_key(self, target_path: str) -> None:
        """
        Export the public key of the cryptosystem.
        """
        # store the secret_key to restore it later
        temp_context = self.context.serialize(save_public_key=True, save_secret_key=True)

        # drop the secret_key from the context
        self.context.make_context_public()
        public_context = self.context.serialize()
        file_utils.write_data(target_path, public_context)
        self.logger.debug(f"Public key saved at {target_path}")

        # restore the secret_key to the context
        self.context = ts.context_from(temp_context)
        self.logger.debug("Secret key restored.")

    def export_private_key(self, target_path: str) -> None:
        """
        Export the private key of the cryptosystem.
        Args:
            target_path (str): The path to save the private key.
        """
        secret_context = self.context.serialize(save_secret_key=True)
        file_utils.write_data(target_path, secret_context)
        self.logger.debug(f"Private key saved at {target_path}")

    def __represent(self, img_path: str) -> List[List[float]]:
        """
        Represent a facial image to vector embedding
        Args:
            img_path (str): The path to the image.
        Returns:
            embeddings (list of list of float): vector embeddings of the faces in the image.
        """
        results = DeepFace.represent(
            img_path, model_name=self.model_name, detector_backend=self.face_detector
        )

        embeddings = []
        for result in results:
            embedding = result["embedding"]
            embedding = self.__normalize_embedding(embedding)
            embeddings.append(embedding)

        return embeddings

    def __normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize the embedding vector
        Args:
            embedding (list of float): The vector embedding.
        Returns:
            normalized_embedding (list of float): The normalized vector embedding.
        """
        # VGG-Face embeddings are already normalized and positive
        if self.model_name == "VGG-Face":
            return embedding
        # eucli
        if self.distance_metric == "euclidean":
            return embedding

        # Facenet and Facenet512 embeddings may have negative values, apply min-max
        # pylint: disable=nested-min-max
        if self.model_name == "Facenet":
            min_val = min(-5.06, min(embedding))
            max_val = max(5.04, max(embedding))
        elif self.model_name == "Facenet512":
            min_val = min(-4.78, min(embedding))
            max_val = max(4.79, max(embedding))
        else:
            raise ValueError(f"Unsupported model {self.model_name}")

        for idx, dim_val in enumerate(embedding):
            embedding[idx] = ((np.array(dim_val) - min_val) / (max_val - min_val)).tolist()

        # Facenet and Facenet512 embeddings are not l2 normalized
        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist()

    def __encrypt(self, embedding: List[float]) -> str:
        """
        Encrypt a vector embedding
        Args:
            embedding (list of float): The vector embedding.
        Returns:
            encrypted_embedding (str): The encrypted embedding as base64 string.
        """
        encrypted_embedding_bytes = ts.ckks_vector(self.context, embedding)
        encrypted_embedding_proto = encrypted_embedding_bytes.serialize()
        return base64.b64encode(encrypted_embedding_proto).decode("utf-8")

    def restore(self, encrypted_embedding: str) -> float:
        """
        Decrypt an encrypted embedding
        Args:
            encrypted_embedding (str): The encrypted embedding as base64 string.
        Returns:
            embedding (list of float): The decrypted embedding.
        """
        encrypted_embedding_bytes = base64.b64decode(encrypted_embedding)
        encrypted_embedding_proto = ts.ckks_vector_from(self.context, encrypted_embedding_bytes)
        return encrypted_embedding_proto.decrypt()[0]

    def calculate_encrypted_distance(
        self, encrypted_embedding_alpha: str, encrypted_embedding_beta: str
    ) -> str:
        """
        Calculate the encrypted distance between two encrypted embeddings
        Args:
            encrypted_embedding_alpha (str): The first encrypted embedding as base64 string.
            encrypted_embedding_beta (str): The second encrypted embedding as base64 string.
        Returns:
            encrypted_distance (str): The encrypted distance between the two embeddings
                as base64 string.
        """
        encrypted_embedding_alpha_bytes = base64.b64decode(encrypted_embedding_alpha)
        alpha = ts.ckks_vector_from(self.context, encrypted_embedding_alpha_bytes)

        encrypted_embedding_beta_bytes = base64.b64decode(encrypted_embedding_beta)
        beta = ts.ckks_vector_from(self.context, encrypted_embedding_beta_bytes)

        if self.distance_metric == "euclidean":
            diff = alpha - beta
            encrypted_distance_proto = diff.dot(diff)
        elif self.distance_metric == "cosine":
            one = ts.ckks_vector(self.context, [1])
            one.link_context(self.context)
            encrypted_distance_proto = one - alpha.dot(beta)
        else:
            raise ValueError(f"Unsupported distance metric {self.distance_metric}")

        encrypted_distance_proto = encrypted_distance_proto.serialize()
        return base64.b64encode(encrypted_distance_proto).decode("utf-8")

    def securely_embed(self, img_path: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        """
        Represent a facial image to encrypted vector embedding.
            The both on prem and cloud can use this method to securely embed the image.
            Because it is depending on public key.
        Args:
            img_path (str or list of str): The path to the image(s).
        Returns:
            encrypted_embedding (list of str, or list of list of str): The encrypted embedding
                as base64 string. If input is list, then return type will be list of list of str.
        """
        if isinstance(img_path, list):
            embeddings = []
            for current_img in tqdm(img_path):
                current_embeddings = self.__represent(current_img)
                encrypted_embeddings = [
                    self.__encrypt(embedding) for embedding in current_embeddings
                ]
                embeddings.append(encrypted_embeddings)
            return embeddings

        embeddings = self.__represent(img_path)
        encrypted_embeddings = [self.__encrypt(embedding) for embedding in embeddings]
        return encrypted_embeddings

    def verify(self, plain_distance: float) -> bool:
        """
        Verify the distance between two embeddings
        Args:
            plain_distance (float): The distance between two embeddings.
        Returns:
            is_verified (bool): True if the distance is less than the threshold.
        """
        pivot = {
            "VGG-Face": {
                "euclidean": 1.17,
                "cosine": 0.68,
            },
            "Facenet": {
                "euclidean": 10,
                "cosine": 0.02431508799003538,
            },
            "Facenet512": {
                "euclidean": 23.56,
                "cosine": 0.02232566879533769,
            },
        }

        threshold = pivot.get(self.model_name)[self.distance_metric]

        if self.distance_metric == "euclidean":
            is_verified = plain_distance < threshold**2
        elif self.distance_metric == "cosine":
            is_verified = plain_distance < threshold
        else:
            raise ValueError(f"Unsupported distance metric {self.distance_metric}")

        return is_verified
