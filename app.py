from flask import Flask, render_template, request, jsonify
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
from langchain.llms import OpenAI