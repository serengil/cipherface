# built-in dependencies
import pickle
import base64

# 3rd party dependencies
from lightphe.models.Tensor import EncryptedTensor
from lightphe.models.Homomorphic import Homomorphic


def cast_encrypted_embeddings_to_str(encrypted_embedding: EncryptedTensor) -> str:
    picked_data = pickle.dumps(encrypted_embedding.fractions)
    encrypted_embedding_encoded = base64.b64encode(picked_data).decode("utf-8")
    return encrypted_embedding_encoded


def restore_encrypted_embedding_obj(encrypted_embedding: str, cs: Homomorphic) -> EncryptedTensor:
    decoded_data = base64.b64decode(encrypted_embedding)
    fractions = pickle.loads(decoded_data)
    return EncryptedTensor(fractions=fractions, cs=cs.cs, precision=cs.precision)
