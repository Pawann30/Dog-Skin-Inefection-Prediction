"""
RAG-style Chatbot Engine using TF-IDF for Dog Skin Disease Q&A.
Retrieves the most relevant knowledge chunks and formats answers.
"""

import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DogDiseaseBot:
    def __init__(self, knowledge_path=None):
        if knowledge_path is None:
            knowledge_path = os.path.join(os.path.dirname(__file__), 'disease_data.json')

        with open(knowledge_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.chunks = []
        self.chunk_metadata = []
        self._build_knowledge_base()

        # Build TF-IDF index
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)

    def _build_knowledge_base(self):
        """Parse disease data and chatbot knowledge into searchable chunks."""

        # Process each disease
        for key, disease in self.data['diseases'].items():
            name = disease['name']

            # Overview chunk
            self.chunks.append(f"{name}: {disease['overview']}")
            self.chunk_metadata.append({'source': name, 'type': 'overview'})

            # Symptoms chunk
            symptoms_text = f"Symptoms of {name}: " + ". ".join(disease['symptoms'])
            self.chunks.append(symptoms_text)
            self.chunk_metadata.append({'source': name, 'type': 'symptoms'})

            # Causes chunk
            causes_text = f"Causes of {name}: " + ". ".join(disease['causes'])
            self.chunks.append(causes_text)
            self.chunk_metadata.append({'source': name, 'type': 'causes'})

            # Treatment chunk
            treatment_text = f"Treatment for {name}: " + ". ".join(disease['treatment'])
            self.chunks.append(treatment_text)
            self.chunk_metadata.append({'source': name, 'type': 'treatment'})

            # Prevention chunk
            prevention_text = f"How to prevent {name}: " + ". ".join(disease['prevention'])
            self.chunks.append(prevention_text)
            self.chunk_metadata.append({'source': name, 'type': 'prevention'})

            # Severity and recovery
            extra = (
                f"{name} severity: {disease['severity']}. "
                f"Contagious: {disease['contagious']}. "
                f"Recovery time: {disease['recovery_time']}."
            )
            self.chunks.append(extra)
            self.chunk_metadata.append({'source': name, 'type': 'general_info'})

        # Process chatbot knowledge articles
        for article in self.data.get('chatbot_knowledge', []):
            self.chunks.append(f"{article['topic']}: {article['content']}")
            self.chunk_metadata.append({'source': article['topic'], 'type': 'article'})

    def get_response(self, query, top_k=3):
        """Retrieve the most relevant chunks and compose a response."""
        if not query.strip():
            return "Please ask me a question about dog skin diseases! 🐾"

        # Transform the query
        query_vec = self.vectorizer.transform([query.lower()])

        # Calculate similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k most relevant chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]

        # If best match is too low, return a fallback
        if top_scores[0] < 0.05:
            return (
                "I'm not sure about that specific question. I can help with:\n"
                "• Dog skin diseases (bacterial, fungal, allergic)\n"
                "• Symptoms and causes of skin conditions\n"
                "• Treatment and prevention tips\n"
                "• General skin health and grooming advice\n\n"
                "Try asking something like: *\"How do I treat fungal infections in dogs?\"*"
            )

        # Build response from relevant chunks
        response_parts = []
        seen_sources = set()

        for idx, score in zip(top_indices, top_scores):
            if score < 0.03:
                continue
            chunk = self.chunks[idx]
            meta = self.chunk_metadata[idx]
            source = meta['source']

            if source not in seen_sources:
                seen_sources.add(source)
                # Clean and format the chunk
                response_parts.append(chunk)

        if not response_parts:
            return "I couldn't find relevant information. Please try rephrasing your question."

        # Combine responses
        combined = "\n\n".join(response_parts[:3])

        # Add a helpful footer
        if len(combined) > 800:
            combined = combined[:800] + "..."

        return combined


# Singleton instance
_bot_instance = None


def get_bot():
    """Get or create the chatbot singleton."""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = DogDiseaseBot()
    return _bot_instance


def chat(query):
    """Main entry point for chatbot queries."""
    bot = get_bot()
    return bot.get_response(query)
