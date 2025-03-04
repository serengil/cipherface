# built-in dependencies
from typing import List, Union, Optional
from abc import ABC, abstractmethod

# 3rd party dependencies
from tqdm import tqdm
from deepface import DeepFace
import numpy as np
from lightphe.models.Tensor import EncryptedTensor


class CipherFaceInterface(ABC):
    model_name: str
    face_detector: str
    distance_metric: Optional[str] = None

    @abstractmethod
    def export_public_key(self, target_path: str) -> None:
        pass

    @abstractmethod
    def export_private_key(self, target_path: str) -> None:
        pass

    @abstractmethod
    def encrypt(self, embedding: List[float]) -> str:
        pass

    @abstractmethod
    def restore(self, encrypted_embedding: str) -> Union[float, EncryptedTensor]:
        pass

    @abstractmethod
    def encrypted_compare(self, alpha: str, beta: Union[str, List[float]]) -> str:
        pass

    @abstractmethod
    def verify(self, plain_distance: float) -> bool:
        pass

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
                current_embeddings = self.represent(current_img)
                encrypted_embeddings = [self.encrypt(embedding) for embedding in current_embeddings]
                embeddings.append(encrypted_embeddings)
            return embeddings

        embeddings = self.represent(img_path)
        encrypted_embeddings = [self.encrypt(embedding) for embedding in embeddings]
        return encrypted_embeddings

    def represent(self, img_path: str) -> List[List[float]]:
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
            assert isinstance(result, dict)
            embedding = result["embedding"]
            embedding = self.normalize_embedding(embedding)
            embeddings.append(embedding)

        return embeddings

    def normalize_embedding(self, embedding: List[float]) -> List[float]:
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
        # euclidean distance metric doesn't require positive values
        if self.distance_metric is not None and self.distance_metric == "euclidean":
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
