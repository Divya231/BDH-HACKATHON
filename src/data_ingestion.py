"""
Data Ingestion Module with Pathway Framework
Handles loading and processing of novels and backstories
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
import pathway as pw
from sentence_transformers import SentenceTransformer


class PathwayDataIngestion:
    """
    Data ingestion system using Pathway framework for efficient
    document processing and vector storage
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['paths']['data_dir'])
        self.raw_dir = Path(config['paths']['raw_data'])
        self.processed_dir = Path(config['paths']['processed_data'])
        
        # Pathway configuration
        self.use_vector_store = config['pathway']['use_vector_store']
        self.embedding_model_name = config['pathway']['embedding_model']
        self.top_k = config['pathway']['top_k_retrieval']
        
        # Initialize embedding model if using vector store
        if self.use_vector_store:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Text processing parameters
        self.chunk_size = config['data']['chunk_size']
        self.chunk_overlap = config['data']['chunk_overlap']
        self.max_novel_length = config['data']['max_novel_length']
        
        logger.info("PathwayDataIngestion initialized")
    
    def load_novel(self, novel_path: str) -> str:
        """Load and preprocess a novel text file"""
        try:
            with open(novel_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Basic preprocessing
            text = text.strip()
            words = text.split()
            
            # Truncate if too long
            if len(words) > self.max_novel_length:
                logger.warning(f"Novel exceeds max length. Truncating from {len(words)} to {self.max_novel_length} words")
                text = ' '.join(words[:self.max_novel_length])
            
            logger.info(f"Loaded novel: {len(words)} words")
            return text
            
        except Exception as e:
            logger.error(f"Error loading novel {novel_path}: {e}")
            raise
    
    def load_backstory(self, backstory_path: str) -> str:
        """Load backstory text file"""
        try:
            with open(backstory_path, 'r', encoding='utf-8') as f:
                backstory = f.read().strip()
            
            logger.info(f"Loaded backstory: {len(backstory.split())} words")
            return backstory
            
        except Exception as e:
            logger.error(f"Error loading backstory {backstory_path}: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for processing
        Maintains context across chunks
        """
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += (self.chunk_size - self.chunk_overlap)
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def create_pathway_schema(self, data: List[Dict]) -> pw.Table:
        """
        Create Pathway table from data
        Track B: Using Pathway for document management
        """
        try:
            # Convert to Pathway table
            df = pd.DataFrame(data)
            table = pw.debug.table_from_pandas(df)
            
            logger.info(f"Created Pathway table with {len(data)} rows")
            return table
            
        except Exception as e:
            logger.error(f"Error creating Pathway table: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence transformer"""
        if not self.use_vector_store:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def setup_vector_store(self, chunks: List[str], metadata: List[Dict]) -> Dict:
        """
        Setup Pathway vector store for semantic search
        Critical for long-context retrieval
        """
        if not self.use_vector_store:
            logger.warning("Vector store disabled in config")
            return {}
        
        logger.info("Setting up Pathway vector store")
        
        # Generate embeddings
        embeddings = self.create_embeddings(chunks)
        
        # Create vector store data
        vector_data = []
        for i, (chunk, emb, meta) in enumerate(zip(chunks, embeddings, metadata)):
            vector_data.append({
                'id': i,
                'text': chunk,
                'embedding': emb,
                'metadata': meta
            })
        
        # Create Pathway table for vector store
        vector_table = self.create_pathway_schema(vector_data)
        
        logger.info(f"Vector store created with {len(vector_data)} entries")
        return {
            'table': vector_table,
            'data': vector_data
        }
    
    def semantic_search(self, query: str, vector_store: Dict, top_k: int = None) -> List[Dict]:
        """
        Perform semantic search over novel chunks
        Returns most relevant passages for a query
        """
        if not self.use_vector_store or not vector_store:
            logger.warning("Vector store not available for search")
            return []
        
        top_k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Compute similarity scores
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        vector_data = vector_store['data']
        embeddings = np.array([item['embedding'] for item in vector_data])
        
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Get top-k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': vector_data[idx]['text'],
                'score': float(similarities[idx]),
                'metadata': vector_data[idx]['metadata']
            })
        
        logger.info(f"Retrieved {len(results)} relevant passages")
        return results
    
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load complete dataset from storage
        Expects dataset with novel files and backstory files
        """
        logger.info(f"Loading dataset from {dataset_path}")
        
        dataset = []
        
        # Implementation depends on your dataset structure
        # This is a template - adjust based on actual data format
        
        # Example structure:
        # dataset/
        #   train/
        #     novel_1.txt
        #     backstory_1.txt
        #     labels.json
        
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return []
        
        # Load metadata/labels if available
        labels_file = dataset_path / "labels.json"
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                labels_data = json.load(f)
        else:
            labels_data = {}
        
        # Find all novel-backstory pairs
        novel_files = sorted(dataset_path.glob("*_novel.txt"))
        
        for novel_file in novel_files:
            story_id = novel_file.stem.replace("_novel", "")
            backstory_file = dataset_path / f"{story_id}_backstory.txt"
            
            if not backstory_file.exists():
                logger.warning(f"Missing backstory for {story_id}")
                continue
            
            # Load texts
            novel_text = self.load_novel(str(novel_file))
            backstory_text = self.load_backstory(str(backstory_file))
            
            # Get label if available
            label = labels_data.get(story_id, None)
            
            dataset.append({
                'story_id': story_id,
                'novel': novel_text,
                'backstory': backstory_text,
                'label': label,
                'novel_file': str(novel_file),
                'backstory_file': str(backstory_file)
            })
        
        logger.info(f"Loaded {len(dataset)} examples")
        return dataset
    
    def process_example(self, example: Dict) -> Dict:
        """
        Process a single example for BDH model
        Creates chunks, embeddings, and prepares for classification
        """
        story_id = example['story_id']
        logger.info(f"Processing example: {story_id}")
        
        # Chunk the novel
        novel_chunks = self.chunk_text(example['novel'])
        
        # Create metadata for each chunk
        chunk_metadata = [
            {
                'story_id': story_id,
                'chunk_id': i,
                'total_chunks': len(novel_chunks),
                'type': 'novel'
            }
            for i in range(len(novel_chunks))
        ]
        
        # Setup vector store for this novel
        vector_store = self.setup_vector_store(novel_chunks, chunk_metadata)
        
        # Extract key claims from backstory (optional enhancement)
        backstory_chunks = self.chunk_text(example['backstory'])
        
        processed = {
            'story_id': story_id,
            'novel_chunks': novel_chunks,
            'backstory': example['backstory'],
            'backstory_chunks': backstory_chunks,
            'label': example.get('label'),
            'vector_store': vector_store,
            'metadata': {
                'num_novel_chunks': len(novel_chunks),
                'num_backstory_chunks': len(backstory_chunks),
                'novel_length': len(example['novel'].split()),
                'backstory_length': len(example['backstory'].split())
            }
        }
        
        return processed
    
    def save_processed_data(self, processed_data: List[Dict], output_path: str):
        """Save processed data for later use"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save without vector store (too large)
        save_data = []
        for item in processed_data:
            save_item = {k: v for k, v in item.items() if k != 'vector_store'}
            save_data.append(save_item)
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Saved processed data to {output_path}")


def main():
    """Test data ingestion"""
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize ingestion
    ingestion = PathwayDataIngestion(config)
    
    # Test loading (adjust paths as needed)
    dataset = ingestion.load_dataset(config['paths']['raw_data'])
    
    if dataset:
        # Process first example
        processed = ingestion.process_example(dataset[0])
        logger.info(f"Processed example metadata: {processed['metadata']}")


if __name__ == "__main__":
    main()
