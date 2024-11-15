import json
from sqlalchemy import create_engine, Column, Integer, String, Table, ForeignKey, Float, Boolean
from sqlalchemy.orm import declarative_base, relationship
from database import Base


# Hierarchy table to manage parent-child relationships between goals
class Hierarchy(Base):
    __tablename__ = "hierarchy"

    id = Column(Integer, primary_key=True)
    refinement = Column(String(3))
    high_level_goal_id = Column(Integer, ForeignKey('goal.id'))
    subgoal_id = Column(Integer, ForeignKey('goal.id'), nullable=False)

    high_level_goal = relationship("Goal", foreign_keys=[high_level_goal_id], back_populates="high_level_hierarchies")
    subgoal = relationship("Goal", foreign_keys=[subgoal_id], back_populates="subgoal_hierarchies")


class Goal(Base):
    __tablename__ = "goal"

    id = Column(Integer, primary_key=True)
    goal_type = Column(String(10))
    goal_name = Column(String(100))

    outputs = relationship("Outputs", back_populates="goal", cascade="all, delete-orphan") # one-to-many relationship`

    triple_filtered = relationship("Triple_Filtered", back_populates="goal",
                                   foreign_keys="[Triple_Filtered.high_level_goal_id]", cascade="all, delete-orphan")
    triple_filtered_subgoals = relationship("Triple_Filtered", back_populates="subgoal",
                                            foreign_keys="[Triple_Filtered.subgoal_id]", cascade="all, delete-orphan")

    high_level_hierarchies = relationship("Hierarchy", foreign_keys="[Hierarchy.high_level_goal_id]",
                                          back_populates="high_level_goal", cascade="all, delete-orphan")
    subgoal_hierarchies = relationship("Hierarchy", foreign_keys="[Hierarchy.subgoal_id]", back_populates="subgoal",
                                       cascade="all, delete-orphan")

    param = relationship("Exploration_Parameter", back_populates="goal", foreign_keys="[Exploration_Parameter.goal_id]", cascade="all, delete-orphan")


class Outputs(Base):
    __tablename__ = "outputs"

    id = Column(Integer, primary_key=True)
    generated_text = Column(String(255))
    goal_type = Column(String(10))
    entailed_triple = Column(String(100))
    goal_id = Column(Integer, ForeignKey('goal.id'))

    goal = relationship("Goal", back_populates="outputs") # many-to-one relationship

    def set_entailed_triple(self, triple_list):
        self.entailed_triple = json.dumps(triple_list)

    def get_entailed_triples(self):
        return json.loads(self.entailed_triple)


class Triple_Filtered(Base):
    __tablename__ = "triple_filtered"

    id = Column(Integer, primary_key=True)
    subgoal_id = Column(Integer, ForeignKey('goal.id'))
    triple_filtered_from_hlg = Column(String(100)) # triple used to create the subgoal
    high_level_goal_id = Column(Integer, ForeignKey('goal.id'))

    goal = relationship("Goal", back_populates="triple_filtered", foreign_keys=[high_level_goal_id])
    subgoal = relationship("Goal", back_populates="triple_filtered_subgoals", foreign_keys=[subgoal_id])

    def set_entailed_triple(self, triple_list):
        self.triple_filtered_from_hlg = json.dumps(triple_list)

    def get_entailed_triples(self):
        return json.loads(self.triple_filtered_from_hlg)


class Exploration_Parameter(Base):
    __tablename__ = "exploration_parameter"
    id = Column(Integer, primary_key=True)
    goal_id = Column(Integer, ForeignKey('goal.id'))
    max_depth = Column(Integer)
    beam_width = Column(Integer)

    goal = relationship("Goal", back_populates="param")