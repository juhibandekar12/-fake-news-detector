from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Depends, Query
from bson import ObjectId
from backend.database import news_collection, users_collection
from backend.auth import get_current_user
from backend.models import NewsPostRequest, NewsPostResult, NewsResponse
from ai.toxicity import check_toxicity
from ai.duplicate import check_duplicate
from ai.predict import predict

router = APIRouter(prefix="/news", tags=["News"])


def serialize_news(doc: dict) -> dict:
    """Convert a MongoDB news document to a JSON-serializable dict."""
    return {
        "id": str(doc["_id"]),
        "title": doc["title"],
        "content": doc["content"],
        "author_id": doc.get("author_id", ""),
        "author_name": doc.get("author_name", "Anonymous"),
        "category": doc.get("category", "General"),
        "is_real": doc.get("is_real", True),
        "bert_confidence": doc.get("bert_confidence", 0),
        "fake_prob": doc.get("fake_prob", 0),
        "real_prob": doc.get("real_prob", 0),
        "upvotes": doc.get("upvotes", 0),
        "downvotes": doc.get("downvotes", 0),
        "is_auto_collected": doc.get("is_auto_collected", False),
        "created_at": doc.get("created_at", datetime.now(timezone.utc)),
    }


@router.post("/post", response_model=NewsPostResult)
async def post_news(request: NewsPostRequest, current_user: dict = Depends(get_current_user)):
    """Submit news through the full moderation pipeline."""
    full_text = f"{request.title} {request.content}"

    # Step 1: Toxicity check
    toxicity_result = check_toxicity(full_text)
    if toxicity_result["is_toxic"]:
        return {
            "status": "rejected",
            "message": toxicity_result["message"],
        }

    # Step 2: Duplicate check
    duplicate_result = await check_duplicate(request.content)
    if duplicate_result["is_duplicate"]:
        return {
            "status": "duplicate",
            "message": f"This article is {duplicate_result['similarity']}% similar to: {duplicate_result['matched_title']}",
            "similarity": duplicate_result["similarity"],
        }

    # Step 3: BERT classification
    prediction = predict(full_text)

    # ── CROSS-CHECK WITH NEWS API / WEB SEARCH ──
    from services.fact_checker import cross_reference_news
    verification = cross_reference_news(request.title)
    
    is_real = prediction["is_real"]
    final_confidence = prediction["confidence"]
    
    # ── Override BERT if NewsAPI/WebSearch verifies the news ──
    if not is_real and verification["is_verified"]:
        is_real = True
        final_confidence = 90.0  # High confidence because actual web source matched
        # Optionally tag the category based on verification source
        request.category = verification.get("source", request.category)
    elif is_real and verification["is_verified"]:
        final_confidence = max(final_confidence, 85.0)

    # ── Handle FAKE NEWS ──
    if not is_real:
        # Store as fake (is_real=False) so admin can see rejected
        news_doc = {
            "title": request.title,
            "content": request.content,
            "author_id": str(current_user["_id"]),
            "author_name": current_user["username"],
            "category": request.category,
            "is_real": False,
            "bert_confidence": final_confidence,
            "fake_prob": prediction["fake_prob"],
            "real_prob": prediction["real_prob"],
            "upvotes": 0,
            "downvotes": 0,
            "is_auto_collected": False,
            "created_at": datetime.now(timezone.utc),
        }
        await news_collection.insert_one(news_doc)

        # Decrease trust score
        await users_collection.update_one(
            {"_id": current_user["_id"]},
            {"$inc": {"trust_score": -10}},
        )

        return {
            "status": "fake",
            "message": "FAKE NEWS",
            "is_real": False,
            "confidence": final_confidence,
            "real_prob": prediction["real_prob"],
            "fake_prob": prediction["fake_prob"],
        }

    # ── Handle REAL NEWS ──
    # Real news — save to DB
    news_doc = {
        "title": request.title,
        "content": request.content,
        "author_id": str(current_user["_id"]),
        "author_name": current_user["username"],
        "category": request.category,
        "is_real": True,
        "bert_confidence": final_confidence,
        "fake_prob": prediction["fake_prob"],
        "real_prob": prediction["real_prob"],
        "upvotes": 0,
        "downvotes": 0,
        "is_auto_collected": False,
        "created_at": datetime.now(timezone.utc),
    }
    result = await news_collection.insert_one(news_doc)

    # Increase trust score
    await users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$inc": {"trust_score": 1}},
    )

    return {
        "status": "approved",
        "message": "REAL NEWS",
        "is_real": True,
        "confidence": final_confidence,
        "real_prob": prediction["real_prob"],
        "fake_prob": prediction["fake_prob"],
        "news_id": str(result.inserted_id),
    }


@router.get("/feed")
async def get_feed(page: int = Query(1, ge=1), limit: int = Query(10, ge=1, le=50)):
    """Get paginated news feed (only real/approved news)."""
    skip = (page - 1) * limit
    cursor = news_collection.find({"is_real": True}).sort("created_at", -1).skip(skip).limit(limit)
    articles = []
    async for doc in cursor:
        articles.append(serialize_news(doc))

    total = await news_collection.count_documents({"is_real": True})
    return {
        "articles": articles,
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit,
    }


@router.get("/trending")
async def get_trending():
    """Get top 10 articles by upvotes."""
    cursor = news_collection.find({"is_real": True}).sort("upvotes", -1).limit(10)
    articles = []
    async for doc in cursor:
        articles.append(serialize_news(doc))
    return articles


@router.get("/breaking")
async def get_breaking():
    """Get latest 10 articles."""
    cursor = news_collection.find({"is_real": True}).sort("created_at", -1).limit(10)
    articles = []
    async for doc in cursor:
        articles.append(serialize_news(doc))
    return articles


@router.get("/search")
async def search_news(query: str = Query(..., min_length=1)):
    """Search news by text query."""
    cursor = news_collection.find(
        {"$text": {"$search": query}, "is_real": True}
    ).sort("created_at", -1).limit(20)
    articles = []
    async for doc in cursor:
        articles.append(serialize_news(doc))
    return articles


@router.get("/my-posts")
async def get_my_posts(current_user: dict = Depends(get_current_user)):
    """Get all posts by the current user."""
    cursor = news_collection.find(
        {"author_id": str(current_user["_id"])}
    ).sort("created_at", -1)
    articles = []
    async for doc in cursor:
        articles.append(serialize_news(doc))
    return articles


@router.get("/category/{category}")
async def get_by_category(category: str):
    """Get news by category."""
    cursor = news_collection.find(
        {"category": category, "is_real": True}
    ).sort("created_at", -1).limit(20)
    articles = []
    async for doc in cursor:
        articles.append(serialize_news(doc))
    return articles


@router.get("/{news_id}")
async def get_news_detail(news_id: str):
    """Get a single news article by ID."""
    try:
        doc = await news_collection.find_one({"_id": ObjectId(news_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid news ID format")

    if not doc:
        raise HTTPException(status_code=404, detail="Article not found")

    return serialize_news(doc)

@router.delete("/{news_id}")
async def delete_news(news_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a news article (must be author or admin)."""
    try:
        obj_id = ObjectId(news_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid news ID format")

    doc = await news_collection.find_one({"_id": obj_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Article not found")

    if doc.get("author_id") != str(current_user["_id"]) and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete this article")

    await news_collection.delete_one({"_id": obj_id})
    return {"status": "success", "message": "Article deleted successfully"}
