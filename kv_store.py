# Design an in-memory key-value store class in Python.
# It supports get(key), set(key, value), and delete(key).
# It also supports transactions:
#   begin() starts a transaction,
#   commit() makes all changes in the current transaction permanent,
#   and rollback() discards them. Transactions can be nested.


class KVStore:
    def __init__(self):
        self.data = dict()

        self.trans_list = list()


    def begin(self):
        self.trans_list.append({})

    def rollback(self):
        if len(self.trans_list) == 0:
            raise Exception("No trans list")

        self.trans_list.pop()

    def commit_to_prim(self, trans):
        if len(trans) == 0:
            return

        for key, v in trans.items():
            if v[0]:
                if key in self.data:
                    self.data.pop(key)
            else:
                self.data[key] = v[1]

    def commit(self):
        if len(self.trans_list) == 0:
            raise Exception("No trans list")

        commit_trans = self.trans_list.pop()

        if len(self.trans_list) == 0:
            self.commit_to_prim(commit_trans)
            return

        last_trans = self.trans_list[-1]

        for key, value in commit_trans.items():
            last_trans[key] = value

    def set(self, key, value):
        if len(self.trans_list) == 0:
            self.data[key] = value
            return

        last_trans = self.trans_list[-1]
        last_trans[key] = (False, value)

    def delete(self, key):
        if len(self.trans_list) == 0:
            if key in self.data:
                self.data.pop(key)
            return

        last_trans = self.trans_list[-1]
        last_trans[key] = (True, -1)


    def get_from_prim(self, key):
        if key in self.data:
            return self.data[key]

        return None

    def get(self, key):
        if len(self.trans_list) == 0:
            return self.get_from_prim(key)

        for i in range(len(self.trans_list)-1, -1, -1):
            trans = self.trans_list[i]
            if key in trans:
                return trans[key][1] if not trans[key][0] else None

        return self.get_from_prim(key)



kv = KVStore()
kv.set("a", 1)
kv.begin()
kv.set("a", 2)
kv.delete("a")
print(kv.get("a"))        # ?
kv.begin()
kv.set("a", 3)
kv.commit()
print(kv.get("a"))        # ?
kv.rollback()
print(kv.get("a"))        # ?