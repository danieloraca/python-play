from surrealdb import SurrealDB, GeometryPoint, Table

with SurrealDB(url="http://127.0.0.1:8000") as db:
    db.use("test", "test")
    auth_token = db.sign_in(username="dan", password="dan")
    db.authenticate(auth_token)
    person = db.create(Table("persons"), {
        "Name": "John",
        "Surname": "Doe",
        "Location": GeometryPoint(-0.11, 22.00),
    })
