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
        self._lock = threading.Lock()
        self._connect()           # première connexion

    # ---------- Azure token ----------
    def _get_token_struct(self):
        cred = DefaultAzureCredential()
        token = cred.get_token("https://database.windows.net/.default").token
        t_bytes = token.encode("utf-16le")
        return struct.pack(f"<I{len(t_bytes)}s", len(t_bytes), t_bytes)

    # ---------- connexion ----------
    def _connect(self):
        self.conn = pyodbc.connect(
            self.conn_string,
            attrs_before={TOKEN_OPT: self._get_token_struct()},
            autocommit=False,  # par sécurité
            timeout=30,
        )

    def _reconnect(self):
        try:
            self.conn.close()
        except Exception:
            pass
        self._connect()

    # ---------- exécution ----------
    def query(self, sql, params=None):
        with self._lock:          # thread-safe
            try:
                cur = self.conn.cursor()
                cur.execute(sql, params or ())
            except pyodbc.Error as e:
                if e.args and e.args[0] == '08S01':
                    self._reconnect()        # token / socket périmés
                    cur = self.conn.cursor()
                    cur.execute(sql, params or ())
                else:
                    raise

            if sql.lstrip().lower().startswith(("insert", "update", "delete")):
                self.conn.commit()
                return None

            cols = [c[0] for c in cur.description]
            data = [dict(zip(cols, row)) for row in cur.fetchall()]
            return data