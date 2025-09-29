from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass

class Vector(Base):
    __tablename__ = 'vectors'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text)
    vector = mapped_column(Vector(1024))
    metadata_: Mapped[dict | None] = mapped_column('metadata', JSONB)

    def __repr__(self):
        return f'Vector(id={self.id}, text={self.text[:50]}..., metadata={self.metadata_})'


