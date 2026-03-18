from __future__ import annotations
from typing import List, Dict, Any
from uuid import UUID
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlmodel import Session, select
from pydantic import BaseModel
from config.database import get_session
from models.restaurant import Restaurant, MenuItem
from models.ingestion import MenuUpload, IngestionStatus, IngestionSource
from services.ingestion import IngestionOrchestrator
from utils.file_handler import save_uploaded_file, FileUploadError


router = APIRouter(prefix="/ingestion", tags=["Menu Ingestion"])


class UploadResponse(BaseModel):
    upload_id: str
    restaurant_id: str
    status: str
    items_created: int
    items_failed: int
    processing_time_seconds: float
    error_message: str | None = None
    notes: str = ""


class RestaurantCreateRequest(BaseModel):
    name: str
    location: str | None = None
    tags: List[str] = []


class UrlScrapeRequest(BaseModel):
    restaurant_id: str
    url: str
    currency: str | None = None


class RestaurantResponse(BaseModel):
    id: str
    name: str
    location: str | None
    tags: List[str]


@router.post("/restaurants", response_model=RestaurantResponse)
def create_restaurant(
    request: RestaurantCreateRequest,
    session: Session = Depends(get_session)
) -> RestaurantResponse:
    if not request.name:
        raise HTTPException(status_code=400, detail="Restaurant name is required")
    
    restaurant = Restaurant(
        name=request.name,
        location=request.location,
        tags=request.tags
    )
    
    session.add(restaurant)
    session.commit()
    session.refresh(restaurant)
    
    return RestaurantResponse(
        id=str(restaurant.id),
        name=restaurant.name,
        location=restaurant.location,
        tags=restaurant.tags
    )


@router.get("/restaurants", response_model=List[RestaurantResponse])
def list_restaurants(session: Session = Depends(get_session)) -> List[RestaurantResponse]:
    statement = select(Restaurant)
    restaurants = session.exec(statement).all()
    
    return [
        RestaurantResponse(
            id=str(r.id),
            name=r.name,
            location=r.location,
            tags=r.tags
        )
        for r in restaurants
    ]


@router.delete("/restaurants/{restaurant_id}/menu")
def delete_restaurant_menu(
    restaurant_id: str,
    session: Session = Depends(get_session)
) -> Dict[str, Any]:
    """Delete all menu items for a specific restaurant."""
    if not restaurant_id:
        raise HTTPException(status_code=400, detail="restaurant_id is required")
    
    try:
        restaurant_uuid = UUID(restaurant_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid restaurant_id format")
    
    restaurant = session.get(Restaurant, restaurant_uuid)
    if not restaurant:
        raise HTTPException(status_code=404, detail=f"Restaurant {restaurant_id} not found")
    
    # Delete all menu items for this restaurant
    statement = select(MenuItem).where(MenuItem.restaurant_id == restaurant_uuid)
    menu_items = session.exec(statement).all()
    
    deleted_count = len(menu_items)
    
    for item in menu_items:
        session.delete(item)
    
    session.commit()
    
    return {
        "restaurant_id": restaurant_id,
        "restaurant_name": restaurant.name,
        "deleted_items": deleted_count,
        "message": f"Successfully deleted {deleted_count} menu items"
    }


@router.post("/upload/pdf", response_model=UploadResponse)
async def upload_pdf_menu(
    restaurant_id: str = Form(...),
    file: UploadFile = File(...),
    currency: str | None = Form(None),
    session: Session = Depends(get_session)
) -> UploadResponse:
    if not restaurant_id:
        raise HTTPException(status_code=400, detail="restaurant_id is required")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        restaurant_uuid = UUID(restaurant_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid restaurant_id format")
    
    restaurant = session.get(Restaurant, restaurant_uuid)
    if not restaurant:
        raise HTTPException(status_code=404, detail=f"Restaurant {restaurant_id} not found")
    
    try:
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="File is empty")
        
        file_path = save_uploaded_file(file_content, file.filename)
        
    except FileUploadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    orchestrator = IngestionOrchestrator()
    
    try:
        upload = orchestrator.process_pdf_upload(
            session=session,
            restaurant_id=restaurant_uuid,
            file_path=file_path,
            original_filename=file.filename,
            currency=currency
        )
        
        notes = ""
        if upload.status == IngestionStatus.COMPLETED:
            notes = f"Successfully extracted {upload.items_created} menu items"
        
        return UploadResponse(
            upload_id=str(upload.id),
            restaurant_id=str(upload.restaurant_id),
            status=upload.status.value if isinstance(upload.status, IngestionStatus) else upload.status,
            items_created=upload.items_created,
            items_failed=upload.items_failed,
            processing_time_seconds=upload.processing_time_seconds or 0.0,
            error_message=upload.error_message,
            notes=notes
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/uploads/{upload_id}", response_model=Dict[str, Any])
def get_upload_status(
    upload_id: str,
    session: Session = Depends(get_session)
) -> Dict[str, Any]:
    if not upload_id:
        raise HTTPException(status_code=400, detail="upload_id is required")
    
    try:
        upload_uuid = UUID(upload_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid upload_id format")
    
    upload = session.get(MenuUpload, upload_uuid)
    if not upload:
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found")
    
    return {
        "upload_id": str(upload.id),
        "restaurant_id": str(upload.restaurant_id),
        "status": upload.status.value if isinstance(upload.status, IngestionStatus) else upload.status,
        "source_type": upload.source_type.value if isinstance(upload.source_type, IngestionSource) else upload.source_type,
        "items_created": upload.items_created,
        "items_failed": upload.items_failed,
        "processing_time_seconds": upload.processing_time_seconds,
        "error_message": upload.error_message,
        "created_at": upload.created_at.isoformat(),
        "updated_at": upload.updated_at.isoformat(),
        "parsed_data": upload.parsed_data
    }


@router.get("/uploads", response_model=List[Dict[str, Any]])
def list_uploads(
    restaurant_id: str | None = None,
    session: Session = Depends(get_session)
) -> List[Dict[str, Any]]:
    statement = select(MenuUpload)
    
    if restaurant_id:
        try:
            restaurant_uuid = UUID(restaurant_id)
            statement = statement.where(MenuUpload.restaurant_id == restaurant_uuid)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid restaurant_id format")
    
    uploads = session.exec(statement).all()
    
    return [
        {
            "upload_id": str(u.id),
            "restaurant_id": str(u.restaurant_id),
            "status": u.status.value if isinstance(u.status, IngestionStatus) else u.status,
            "source_type": u.source_type.value if isinstance(u.source_type, IngestionSource) else u.source_type,
            "items_created": u.items_created,
            "items_failed": u.items_failed,
            "original_filename": u.original_filename,
            "processing_time_seconds": u.processing_time_seconds,
            "created_at": u.created_at.isoformat()
        }
        for u in uploads
    ]


@router.post("/upload/url", response_model=UploadResponse)
async def upload_url_menu(
    request: UrlScrapeRequest,
    session: Session = Depends(get_session),
) -> UploadResponse:
    if not request.restaurant_id:
        raise HTTPException(status_code=400, detail="restaurant_id is required")

    if not request.url:
        raise HTTPException(status_code=400, detail="url is required")

    try:
        restaurant_uuid = UUID(request.restaurant_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid restaurant_id format")

    restaurant = session.get(Restaurant, restaurant_uuid)
    if not restaurant:
        raise HTTPException(
            status_code=404, detail=f"Restaurant {request.restaurant_id} not found"
        )

    orchestrator = IngestionOrchestrator()

    try:
        upload = await orchestrator.process_url_upload_async(
            session=session,
            restaurant_id=restaurant_uuid,
            source_url=request.url,
            currency=request.currency,
        )

        notes = ""
        if upload.status == IngestionStatus.COMPLETED:
            notes = f"Successfully scraped and extracted {upload.items_created} menu items from {request.url}"

        return UploadResponse(
            upload_id=str(upload.id),
            restaurant_id=str(upload.restaurant_id),
            status=upload.status.value
            if isinstance(upload.status, IngestionStatus)
            else upload.status,
            items_created=upload.items_created,
            items_failed=upload.items_failed,
            processing_time_seconds=upload.processing_time_seconds or 0.0,
            error_message=upload.error_message,
            notes=notes,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
