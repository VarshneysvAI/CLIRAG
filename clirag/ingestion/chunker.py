import re
from typing import List

class RecursiveTextChunker:
    """
    Subdivides large text corpora into smaller overlapping chunks
    to avoid LLM/Embedding maximum token context limits.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        :param chunk_size: Approximate maximum number of words per chunk.
        :param overlap: Number of words to overlap between consecutive chunks to preserve semantic boundary context.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        """
        Splits text recursively. 
        First, we try splitting by double newlines (paragraphs). 
        If a paragraph is still too large, we fall back to word-level splitting.
        """
        if not text or not text.strip():
            return []

        # 1. Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk_words = []
        
        for paragraph in paragraphs:
            para_words = paragraph.split()
            
            # If the paragraph alone is larger than our chunk size, we must split it by words
            if len(para_words) > self.chunk_size:
                # Flush existing buffer first
                if current_chunk_words:
                    chunks.append(" ".join(current_chunk_words))
                    current_chunk_words = current_chunk_words[-self.overlap:] if self.overlap > 0 else []
                
                # Slide over the massive paragraph
                for i in range(0, len(para_words), self.chunk_size - self.overlap):
                    slice_end = min(i + self.chunk_size, len(para_words))
                    word_slice = para_words[i:slice_end]
                    chunks.append(" ".join(word_slice))
                
                # The last overlapping tail becomes the seed for the next paragraph
                current_chunk_words = para_words[-(self.overlap):] if self.overlap > 0 else []
                
            else:
                # If adding this paragraph exceeds the chunk size, flush the buffer
                if current_chunk_words and (len(current_chunk_words) + len(para_words) > self.chunk_size):
                    chunks.append(" ".join(current_chunk_words))
                    # Carry over overlap
                    current_chunk_words = current_chunk_words[-self.overlap:] if self.overlap > 0 else []
                
                current_chunk_words.extend(para_words)
        
        # Flush whatever is left
        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
            
        return chunks

if __name__ == "__main__":
    print("\n--- Recursive Text Chunker Test ---")
    chunker = RecursiveTextChunker(chunk_size=10, overlap=3)
    sample_text = "This is a very long string that we need to test. \n\nIt has multiple paragraphs. We want to ensure that overlapping chunk logic works perfectly when dealing with documents."
    chunks = chunker.split_text(sample_text)
    
    for i, c in enumerate(chunks):
        print(f"Chunk {i+1} (Words: {len(c.split())}): {c}")
    print("--- Success ---\n")
