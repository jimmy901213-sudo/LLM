from update_vectorstore import VectorstoreUpdater

updater = VectorstoreUpdater()
all_docs = updater.vectorstore.get()
print('keys:', list(all_docs.keys()))
metas = all_docs.get('metadatas')
ids = all_docs.get('ids')
if not metas:
    print('No metadatas found')
else:
    for i, m in enumerate(metas[:10]):
        print(i, 'id=', ids[i] if ids and i < len(ids) else None)
        print(m)
