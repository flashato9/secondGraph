import lancedb

db = lancedb.connect("./.agent_data/vectors")
table_name = "chat_history"

# Drop the old, broken table
if table_name in db.table_names():
    db.drop_table(table_name)

# Create the table with the correct dimension (3072)
# Tip: Use [0.0] * 3072 if you're using the high-res model
db.create_table(
    table_name, 
    data=[{"text": "seed", "vector": [0.0] * 3072}] 
)
print("Table recreated with 3072 dimensions.")