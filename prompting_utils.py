import os


def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    # TODO
    schema = {}
    with open(schema_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        table_match = re.match(r'(\w+)\((.*?)\)', line)
        if table_match:
            table_name = table_match.group(1)
            columns = [col.strip() for col in table_match.group(2).split(',')]
            schema[table_name] = columns
    return schema

def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    # TODO
    if not response:
        return ""
    code_block = re.search(r"```sql(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if code_block:
        sql = code_block.group(1).strip()
    else:
        match = re.search(r"(SELECT .*?;)", response, re.DOTALL | re.IGNORECASE)
        sql = match.group(1).strip() if match else response.strip()
    sql = sql.replace("\n", " ").strip()
    return sql

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    #with open(output_path, "w") as f:
        #f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em:.4f}\n")
        f.write(f"Record EM: {record_em:.4f}\n")
        f.write(f"Record F1: {record_f1:.4f}\n")
        f.write("\nModel Error Messages:\n")
        if isinstance(error_msgs, (list, dict)):
            f.write(json.dumps(error_msgs, indent=2))
        else:
            f.write(str(error_msgs))