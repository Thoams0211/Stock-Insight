from neo4j import GraphDatabase
from tqdm import tqdm

TXT_PATH = "/root/autodl-tmp/MindMap/data/Fin/relNing.txt"

def sanitize_relationship_type(rel_type):
        # 将关系类型转换为有效的标识符，例如用下划线替换空格和其他字符
        return "REL_" + ''.join(['_' if not c.isalnum() else c for c in rel_type])

class Neo4jHandler:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_relationship(self, entity1, relationship, entity2):
        with self.driver.session() as session:
            session.write_transaction(self._create_and_return_relationship, entity1, relationship, entity2)

    @staticmethod
    def _create_and_return_relationship(tx, entity1, relationship, entity2):
        sanitized_relationship_type = sanitize_relationship_type(relationship)
        query = (
            "MERGE (a:Entity {name: $entity1}) "
            "MERGE (b:Entity {name: $entity2}) "
            f'MERGE (a)-[r: {sanitized_relationship_type}]->(b) '
            "RETURN a, b, r"
        )
        tx.run(query, entity1=entity1, entity2=entity2)

def read_triples_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        triples = [line.strip().split("##") for line in file.readlines()]
    return triples



def main(txtPath):
    uri = "bolt://localhost:7687"  # 你的Neo4j数据库地址
    user = "neo4j"  # 你的Neo4j用户名
    password = "12345678"  # 你的Neo4j密码

    neo4j_handler = Neo4jHandler(uri, user, password)
    file_path = txtPath  # 你的TXT文件路径

    triples = read_triples_from_file(file_path)
    for cnt in tqdm(range(len(triples)), desc="inserting"):
        if len(triples[cnt]) < 3:
            continue
        entity1 = triples[cnt][0]
        relationship = triples[cnt][1]
        entity2 = triples[cnt][2]
        try:
            neo4j_handler.create_relationship(entity1, relationship, entity2)
        except:
            print(f"Error creating relationship: {entity1} - {relationship} - {entity2}")
            raise ValueError 
    neo4j_handler.close()

if __name__ == "__main__":
    main(TXT_PATH)
