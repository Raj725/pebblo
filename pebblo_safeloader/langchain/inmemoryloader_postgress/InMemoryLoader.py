from typing import Any, Dict, Iterator, List, Optional

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class InMemoryLoader(BaseLoader):
    """
    Load In-Memory data into a list of Documents.
    """

    def __init__(
        self,
        texts: List[str],
        *,
        source: Optional[str] = None,
        ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Args:
            texts: List of text data.
            source: Source of the text data.
                Optional. Defaults to None.
            ids: List of unique identifiers for each text.
                Optional. Defaults to None.
            metadata: Metadata for all texts.
                Optional. Defaults to None.
            metadatas: List of metadata for each text.
                Optional. Defaults to None.
        """
        self.texts = texts
        self.source = source
        self.ids = ids
        self.metadata = metadata
        self.metadatas = metadatas

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy load In-Memory data into a list of Documents.

        Returns:
            Iterator of Documents
        """
        for i, text in enumerate(self.texts):
            _id = None
            metadata = self.metadata or {}
            if self.metadatas and i < len(self.metadatas) and self.metadatas[i]:
                metadata.update(self.metadatas[i])
            if self.ids and i < len(self.ids):
                _id = self.ids[i]
            yield Document(id=_id, page_content=text, metadata=metadata)

    def load(self) -> List[Document]:
        """
        Load In-Memory data into a list of Documents.

        Returns:
            List of Documents
        """
        documents = []
        for doc in self.lazy_load():
            documents.append(doc)
        return documents
