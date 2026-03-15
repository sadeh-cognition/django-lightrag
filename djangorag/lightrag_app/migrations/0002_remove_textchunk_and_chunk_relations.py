from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("lightrag_app", "0001_initial"),
    ]

    operations = [
        migrations.DeleteModel(name="RelationChunk"),
        migrations.DeleteModel(name="EntityChunk"),
        migrations.DeleteModel(name="TextChunk"),
    ]
