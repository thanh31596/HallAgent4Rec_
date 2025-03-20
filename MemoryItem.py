import numpy as np
import pandas as pd
import faiss
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import math
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from pydantic import BaseModel, Field
#  Base memory structure (as provided in the notebook)
class MemoryItem(BaseModel):
    content: str
    created_at: datetime
    importance: Optional[float] = 0.0
