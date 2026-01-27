import os, struct, pyodbc, threading
from azure.identity import DefaultAzureCredential

# Laissez le pooling activé
# pyodbc.pooling = True  (par défaut)

TOKEN_OPT = 1256  # SQL_COPT_SS_ACCESS_TOKEN

class DBConnector:
    def __init__(self):
        self.conn_string = os.getenv("DB_CONN_PROD")
        if not self.conn_string:
            raise RuntimeError("DB_CONN_PROD not set")

    # ---------- Azure token ----------
    def _get_token_struct(self):
        cred = DefaultAzureCredential()
        token = cred.get_token("https://database.windows.net/.default").token
        t_bytes = token.encode("utf-16le")
        return struct.pack(f"<I{len(t_bytes)}s", len(t_bytes), t_bytes)

    # ---------- connexion ----------
    def connect(self):
        return pyodbc.connect(
            self.conn_string,
            autocommit=False,  # par sécurité
            timeout=30,
        )

    def _reconnect(self):
        conn = self.connect()
        try:
            conn.close()
        except Exception:
            pass
        return conn

    # ---------- exécution ----------
    def query(self, sql, params=None):
        conn = self.connect()
        with conn.cursor() as cur:
            try:
                cur.execute(sql, params or ())
            except pyodbc.Error as e:
                if e.args and e.args[0] == '08S01':
                    self._reconnect()        # token / socket périmés
                    conn = self.connect()
                    with conn.cursor() as cur:
                        cur.execute(sql, params or ())
                else:
                    raise

            if sql.lstrip().lower().startswith(("insert", "update", "delete")):
                conn.commit()
                return None

            cols = [c[0] for c in cur.description]
            data = [dict(zip(cols, row)) for row in cur.fetchall()]
            return data