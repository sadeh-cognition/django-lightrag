from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("lightrag_app", "0002_remove_textchunk_and_chunk_relations"),
    ]

    operations = [
        migrations.RenameField(
            model_name="documentstatus",
            old_name="chunks_count",
            new_name="documents_count",
        ),
        migrations.RenameField(
            model_name="documentstatus",
            old_name="chunks_list",
            new_name="documents_list",
        ),
    ]
