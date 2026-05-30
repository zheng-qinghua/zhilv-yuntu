from datetime import date as DateType
from pydantic import BaseModel, Field

class TripRequest(BaseModel):
  destination: str = Field(..., description="目的地，例如：大理")
  start_date: DateType = Field(...,description="出发日期")
  end_date: DateType = Field(..., description="结束日期")
  travelers: int = Field(default=1, ge=1, description="人数")
  budget: float = Field(default=5000.0, ge=0, description="总预算（元）")
  preferences: list[str] = Field(default_factory=list, description="偏好标签")
  pace: str = Field(default="适中", description="旅行节奏：轻松 / 适中 / 紧凑")
  special_notes: str | None = Field(default=None, description="额外备注")

class SpotItem(BaseModel):
  """单个景点安排。
  每个 DayPlan 包含 1~N 个 SpotItem。
  """
  name: str
  description: str | None = None
  estimated_cost: float = 0.0
  location: str | None = None

class MealItem(BaseModel):
  name: str
  meal_type: str = '午餐'
  estimated_cost: float = 0.0
  notes: str | None = None

class HotelItem(BaseModel):
  name: str
  level: str | None = None
  estimated_cost: float = 0.0
  location: str | None = None

class TransportItem(BaseModel):
  mode: str = "打车"                          
  from_place: str | None = None
  to_place: str | None = None
  estimated_cost: float = 0.0
  duration: str | None = None

class DayPlan(BaseModel):
  day_index: int = Field(..., ge=1)          
  date: DateType | None = None               
  theme: str | None = None                   
  spots: list[SpotItem] = Field(default_factory=list)
  meals: list[MealItem] = Field(default_factory=list)
  hotel: HotelItem | None = None
  transport: list[TransportItem] = Field(default_factory=list)
  notes: list[str] = Field(default_factory=list)  # 备注列表

class BudgetBreakdown(BaseModel):
  transport: float = 0.0
  hotel: float = 0.0
  meals: float = 0.0
  tickets: float = 0.0
  other: float = 0.0
  total: float = 0.0

class TokenUsage(BaseModel):
  """LLM 调用的 token 消耗统计（用于监控成本）。"""
  rewrite_prompt: int = 0       # Query Rewrite 输入 token
  rewrite_completion: int = 0   # Query Rewrite 输出 token
  embedding_prompt: int = 0     # Embedding 输入 token
  planner_prompt: int = 0       # 行程生成输入 token
  planner_completion: int = 0   # 行程生成输出 token

class Itinerary(BaseModel):
  trip_id: str
  destination: str                           
  summary: str                               
  days: list[DayPlan] = Field(default_factory=list)
  estimated_budget: float = 0.0             
  budget_breakdown: BudgetBreakdown = Field(default_factory=BudgetBreakdown)
  tips: list[str] = Field(default_factory=list)
  source_notes: list[str] = Field(default_factory=list)  
  token_usage: TokenUsage | None = None