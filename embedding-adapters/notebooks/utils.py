import json
from torch import nn
from torch.utils.data import Dataset
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from torch.utils.data import Dataset


def get_document_chunks(pdf_path, model_name="gpt-4o-mini", chunk_size=800, chunk_overlap=400):
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_pages = pdf_loader.load()
    pdf_document = "".join(page.page_content for page in pdf_pages)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    document_chunks = text_splitter.split_text(pdf_document)
    return document_chunks


def load_data(train_path, validation_path):
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(validation_path, 'r') as f:
        validation_data = json.load(f)
    return train_data, validation_data


class LinearAdapter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.linear(x)


class TripletDataset(Dataset):
    def __init__(self, data, base_model, negative_sampler):
        self.data = data
        self.base_model = base_model
        self.negative_sampler = negative_sampler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['question']
        positive = item['chunk']
        negative = self.negative_sampler()
        query_emb = self.base_model.encode(query, convert_to_tensor=True)
        positive_emb = self.base_model.encode(positive, convert_to_tensor=True)
        negative_emb = self.base_model.encode(negative, convert_to_tensor=True)
        return query_emb, positive_emb, negative_emb
