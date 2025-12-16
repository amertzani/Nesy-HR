"""
Documents Store Module
======================

Provides access to stored document metadata.
"""

import json
import os
from typing import List, Dict, Any


def get_all_documents() -> List[Dict[str, Any]]:
    """
    Get all documents from the documents store.
    
    Returns:
        List of document dictionaries
    """
    store_file = "documents_store.json"
    
    if not os.path.exists(store_file):
        return []
    
    try:
        with open(store_file, 'r') as f:
            data = json.load(f)
            return data.get('documents', [])
    except Exception:
        return []


def add_document(
    name: str,
    file_type: str,
    file_path: str,
    size: int = 0
) -> None:
    """
    Add a document to the documents store.
    
    Args:
        name: Document name
        file_type: File extension/type
        file_path: Path to the file
        size: File size in bytes
    """
    store_file = "documents_store.json"
    
    # Load existing documents
    if os.path.exists(store_file):
        try:
            with open(store_file, 'r') as f:
                data = json.load(f)
        except Exception:
            data = {"documents": [], "last_updated": None, "total_documents": 0}
    else:
        data = {"documents": [], "last_updated": None, "total_documents": 0}
    
    # Check if document already exists
    existing = next((d for d in data.get("documents", []) if d.get("name") == name), None)
    
    if existing:
        # Update existing
        existing.update({
            "file_path": file_path,
            "size": size,
            "type": file_type,
            "status": "completed"
        })
    else:
        # Add new document
        from datetime import datetime
        # Get next ID
        existing_ids = [int(d.get("id", "0")) for d in data.get("documents", []) if d.get("id", "0").isdigit()]
        next_id = str(max(existing_ids) + 1) if existing_ids else "1"
        
        doc = {
            "id": next_id,
            "name": name,
            "type": file_type,
            "file_path": file_path,
            "size": size,
            "uploaded_at": datetime.now().isoformat(),
            "status": "completed",
            "facts_extracted": 0
        }
        data["documents"].append(doc)
    
    # Update metadata
    from datetime import datetime
    data["last_updated"] = datetime.now().isoformat()
    data["total_documents"] = len(data.get("documents", []))
    
    # Save
    with open(store_file, 'w') as f:
        json.dump(data, f, indent=2)


def delete_document(name: str = None, document_id: str = None) -> bool:
    """
    Delete a document from the documents store.
    
    Args:
        name: Document name to delete
        document_id: Document ID to delete
    
    Returns:
        True if deleted, False if not found
    """
    store_file = "documents_store.json"
    
    if not os.path.exists(store_file):
        return False
    
    try:
        with open(store_file, 'r') as f:
            data = json.load(f)
    except Exception:
        return False
    
    documents = data.get("documents", [])
    original_count = len(documents)
    
    # Remove document by name or ID
    if name:
        documents = [d for d in documents if d.get("name") != name]
    elif document_id:
        documents = [d for d in documents if d.get("id") != document_id]
    else:
        return False
    
    # Check if anything was removed
    if len(documents) == original_count:
        return False
    
    # Update metadata
    from datetime import datetime
    data["documents"] = documents
    data["last_updated"] = datetime.now().isoformat()
    data["total_documents"] = len(documents)
    
    # Save
    with open(store_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return True

