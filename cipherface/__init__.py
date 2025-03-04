# built-in dependencies
import os
from typing import Optional, List, Union
import base64

# 3rd party dependencies
from deepface import DeepFace
from lightphe import LightPHE
from lightphe.models.Tensor import EncryptedTensor
import tenseal as ts
import numpy as np
from tqdm import tqdm

# project dependencies
from cipherface.commons.logger import Logger
from cipherface.commons import file_utils, phe_utils
from cipherface.interfaces.CipherFaceInterface import CipherFaceInterface


__version__ = "0.0.2"

# pylint: disable=unknown-option-value


class CipherFaceLite(CipherFaceInterface):
    def __init__(
        self,
        model_name: str = "VGG-Face",
        face_detector: str = "opencv",
        algorithm_name: str = "Paillier",
        security_level: int = 80,
        cryptosystem: Optional[str] = None,
    ):
        """
        Build the CipherFaceLite
        Args:
            model_name (str): facial recognition model name.
                Options: VGG-Face, Facenet, Facenet512
            face_detector (str): The path to the face detector.
                Options: opencv, mtcnn, ssd, dlib, retinaface, mediapipe,
                yolov8, yolov11n, yolov11s, yolov11m, yunet, fastmtcnn or centerface
            algorithm_name (str): The name of the algorithm to use for encryption.
                Options: Paillier, Damgard-Jurik, Okamoto-Uchiyama
            security_level (int): The security level of the cryptosystem.
            cryptosystem (str): The path to the cryptosystem. Generates a random
                private-public key pair if None.
        """
        self.logger = Logger(__name__)

        self.model_name = model_name
        self.face_detector = face_detector
        self.algorithm_name = algorithm_name
        assert self.algorithm_name in [
            "Paillier",
            "Damgard-Jurik",
            "Okamoto-Uchiyama",
        ]
        if security_level == 80:
            if algorithm_name != "EllipticCurve-ElGamal":
                key_size = 1024
            else:
                key_size = 160
        elif security_level == 128:
            if algorithm_name != "EllipticCurve-ElGamal":
                key_size = 2048
            else:
                key_size = 224
        elif security_level == 128:
            if algorithm_name != "EllipticCurve-ElGamal":
                key_size = 3072
            else:
                key_size = 256
        elif security_level == 192:
            if algorithm_name != "EllipticCurve-ElGamal":
                key_size = 7680
            else:
                key_size = 384
        else:
            raise ValueError(f"Security level {security_level} not supported.")

        self.cs = LightPHE(
            algorithm_name=algorithm_name, key_size=key_size, precision=19, key_file=cryptosystem
        )
        self.logger.debug(f"{algorithm_name} cryptosystem is just built with {key_size} key size.")

        _ = DeepFace.build_model(model_name=model_name, task="facial_recognition")
        self.logger.debug(f"{model_name} model is just built.")

        _ = DeepFace.build_model(model_name=face_detector, task="face_detector")
        self.logger.debug(f"{face_detector} face detector is just built.")

    def export_public_key(self, target_path: str) -> None:
        """
        Export the public key of the cryptosystem.
        """
        self.cs.export_keys(target_file=target_path, public=True)

    def export_private_key(self, target_path: str) -> None:
        """
        Export the private key of the cryptosystem.
        Args:
            target_path (str): The path to save the private key.
        """
        self.cs.export_keys(target_file=target_path, public=False)

    def encrypt(self, embedding: List[float]) -> str:
        """
        Encrypt a vector embedding
        Args:
            embedding (list of float): The vector embedding.
        Returns:
            encrypted_embedding (str): The encrypted embedding as base64 string.
        """
        encrypted_embedding = self.cs.encrypt(embedding, silent=True)
        assert isinstance(encrypted_embedding, EncryptedTensor)
        encrypted_embedding_str = phe_utils.cast_encrypted_embeddings_to_str(encrypted_embedding)
        return encrypted_embedding_str

    def restore(self, encrypted_embedding: str) -> float:
        """
        Decrypt an encrypted embedding
        Args:
            encrypted_embedding (str): The encrypted embedding as base64 string.
        Returns:
            embedding (list of float): The decrypted embedding.
        """
        encrypted_embedding_obj = phe_utils.restore_encrypted_embedding_obj(
            encrypted_embedding, self.cs
        )
        return self.cs.decrypt(ciphertext=encrypted_embedding_obj)[0]

    def encrypted_compare(self, alpha: str, beta: List[float]) -> str:
        """
        Calculate the encrypted cosine similarity between two encrypted embeddings
        Args:
            alpha (str): The first encrypted embedding as base64 string.
            beta (str): The second encrypted embedding as base64 string.
        Returns:
            encrypted_distance (str): The encrypted distance between the two embeddings
                as base64 string.
        """
        encrypted_embedding = phe_utils.restore_encrypted_embedding_obj(alpha, self.cs)
        encrypted_cosine_similarity = encrypted_embedding @ beta
        return phe_utils.cast_encrypted_embeddings_to_str(encrypted_cosine_similarity)

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
                "cosine": 1 - 0.68,
            },
            "Facenet": {
                "cosine": 1 - 0.02431508799003538,
            },
            "Facenet512": {
                "cosine": 1 - 0.02232566879533769,
            },
        }

        threshold = pivot.get(self.model_name)["cosine"]

        is_verified = plain_distance > threshold
        return is_verified


class CipherFace(CipherFaceInterface):
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
            security_level (int): The security level of the cryptosystem.
                Options are 128, 192 and 256. Default is 128.
                128-bit security level is considered safe until beyond 2030.
                192-bit security level is considered safe until much beyond 2030.
                256-bit security level is considered safe until mucher beyond 2030.
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

        _ = DeepFace.build_model(model_name=model_name, task="facial_recognition")
        self.logger.debug(f"{model_name} model is just built.")

        _ = DeepFace.build_model(model_name=face_detector, task="face_detector")
        self.logger.debug(f"{face_detector} face detector is just built.")

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

    def encrypt(self, embedding: List[float]) -> str:
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

    def encrypted_compare(self, alpha: str, beta: str) -> str:
        """
        Calculate the encrypted cosine or euclideandistance between two encrypted embeddings
        Args:
            alpha (str): The first encrypted embedding as base64 string.
            beta (str): The second encrypted embedding as base64 string.
        Returns:
            encrypted_distance (str): The encrypted distance between the two embeddings
                as base64 string.
        """
        encrypted_embedding_alpha_bytes = base64.b64decode(alpha)
        alpha = ts.ckks_vector_from(self.context, encrypted_embedding_alpha_bytes)

        encrypted_embedding_beta_bytes = base64.b64decode(beta)
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
