# app/database.py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from app.config import settings # Assuming settings.DATABASE_URL is available

DATABASE_URL = settings.DATABASE_URL

engine = create_async_engine(DATABASE_URL) # echo=True for debugging SQL
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    async with engine.begin() as conn:
        # await conn.run_sync(Base.metadata.drop_all) # Optional: drop tables for a clean start during dev
        await conn.run_sync(Base.metadata.create_all)