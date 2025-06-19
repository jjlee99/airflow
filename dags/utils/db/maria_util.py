from airflow.providers.mysql.hooks.mysql import MySqlHook

class DBUtil:
    def __init__(self, conn_id='your_mysql_conn_id'):
        self.conn_id = conn_id

    def execute_query(self, query, params=None, fetch=False):
        """
        쿼리를 실행하고 결과를 반환합니다.
        :param query: 실행할 SQL 쿼리
        :param params: 쿼리 파라미터 (튜플 또는 딕셔너리)
        :param fetch: 결과를 조회해서 반환할지 여부 (True: SELECT 등, False: INSERT/UPDATE/DELETE 등)
        :return: 결과 데이터 (fetch=True일 때)
        """
        hook = MySqlHook(mysql_conn_id=self.conn_id)
        conn = hook.get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            if fetch:
                result = cursor.fetchall()
            else:
                result = None
            conn.commit()
            return result
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

def execute(query, params=None, fetch=False):
    db = DBUtil(conn_id='maria_db_conn')
    results = db.execute_query(query, params=params, fetch=fetch)
    return results

def call_set(key, params=None, fetch=False):
    map = {
        "insertRun":"INSERT INTO TB_AF_RUN(dag_id, run_id) VALUES (%s, %s)"
    }
    execute(map[key], params=params, fetch=fetch)