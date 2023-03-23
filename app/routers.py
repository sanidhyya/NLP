from fastapi import APIRouter

from .models import Sentence
from .nlp.information_extraction import NlpAlgos, KnowledgeGraph
from .handlers.dataset_handler import IE_brand, get_dataframe_head, get_dataframe_sentence

# router object for handling api routes
router = APIRouter()

@router.post("/postagging", response_description="POS Tagging words in a sentence")
async def pos_tag(sentence : Sentence):
    text = sentence.sentence
    pos_applied = NlpAlgos().POS_tagging(text)
    # pos_applied = json.dumps(pos_applied) 
    return pos_applied

@router.post("/dependency-graph", response_description="generates a dependency graph for a sentence")
async def generate_dependency_graph(sentence : Sentence):
    text = sentence.sentence
    dependency_graph = NlpAlgos().dependency_graph(text)
    return dependency_graph


@router.post("/summarize", response_description="generates a text summary for a sentence")
async def generate_summary(sentence : Sentence):
    long_review  = sentence.sentence
    summarized_review = NlpAlgos().summarize(long_review)
    return summarized_review


@router.post("/sentiment", response_description="generates a text summary for a sentence")
async def generate_sentiment(brand : str):
    mean_brand_sentiment = IE_brand(brand)

    mean_brand_sentiment =  "{:.3f}".format(mean_brand_sentiment)

    return {
        "average_brand_sentiment" : mean_brand_sentiment
    }


@router.post("/ner", response_description= "Named entitity recognition for a sentence")
async def generate_ner(sentence : Sentence):
    ner_applied = NlpAlgos().apply_ner(sentence.sentence)
    return ner_applied

@router.post("/reviews/{brand}", response_description= "Named entitity recognition for a sentence")
async def generate_dataframe(brand : str):
    dataframe = get_dataframe_head(brand)
    return dataframe


@router.post("/knowledge-graph", response_description= "Named entitity recognition for a sentence")
async def generate_knowledge_graph(sentence : Sentence):
    knowledge_graph = KnowledgeGraph().knowledge_graph(sentence.sentence)
    return knowledge_graph


@router.post("/sentence", response_description= "Get the top review for a brand")
async def generate_brand_sentence(brand : str):
    review = get_dataframe_sentence(brand)
    return review










