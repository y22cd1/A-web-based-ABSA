"""Add email field to User

Revision ID: 3a4a81891f2
Revises: 40792ad4b727
Create Date: 2025-09-02 23:59:54.792184
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '3a4a81891f2'
down_revision = '40792ad4b727'
branch_labels = None
depends_on = None


def upgrade():
    # Step 1: Add column as nullable
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.add_column(sa.Column('email', sa.String(length=150), nullable=True))

    # Step 2: Backfill existing rows with dummy emails
    conn = op.get_bind()
    users = conn.execute(sa.text("SELECT id, username FROM user")).fetchall()
    for u in users:
        dummy_email = f"{u.username.lower()}@example.com"
        conn.execute(
            sa.text("UPDATE user SET email = :email WHERE id = :id"),
            {"email": dummy_email, "id": u.id}
        )

    # Step 3: Make column non-nullable + unique
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.alter_column('email', nullable=False)
        batch_op.create_unique_constraint("uq_user_email", ['email'])


def downgrade():
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.drop_constraint("uq_user_email", type_="unique")
        batch_op.drop_column('email')
