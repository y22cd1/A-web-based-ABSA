"""Add actual_sentiment to Review

Revision ID: 9f2c7d1a8b6f
Revises: add_accuracy_tables
Create Date: 2026-04-07 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9f2c7d1a8b6f'
down_revision = 'add_accuracy_tables'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('review', schema=None) as batch_op:
        batch_op.add_column(sa.Column('actual_sentiment', sa.String(length=20), nullable=True))


def downgrade():
    with op.batch_alter_table('review', schema=None) as batch_op:
        batch_op.drop_column('actual_sentiment')
