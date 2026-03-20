# Generated migration to remove title field from Document model

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("django_lightrag", "0001_initial"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="document",
            name="title",
        ),
    ]
