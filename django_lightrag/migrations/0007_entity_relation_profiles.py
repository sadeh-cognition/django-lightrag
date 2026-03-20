from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("django_lightrag", "0006_delete_processingjob"),
    ]

    operations = [
        migrations.AddField(
            model_name="entity",
            name="profile_input_hash",
            field=models.CharField(blank=True, max_length=64),
        ),
        migrations.AddField(
            model_name="entity",
            name="profile_key",
            field=models.CharField(blank=True, max_length=255),
        ),
        migrations.AddField(
            model_name="entity",
            name="profile_updated_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="entity",
            name="profile_value",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="relation",
            name="profile_input_hash",
            field=models.CharField(blank=True, max_length=64),
        ),
        migrations.AddField(
            model_name="relation",
            name="profile_key",
            field=models.CharField(blank=True, max_length=255),
        ),
        migrations.AddField(
            model_name="relation",
            name="profile_updated_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="relation",
            name="profile_value",
            field=models.TextField(blank=True),
        ),
    ]
