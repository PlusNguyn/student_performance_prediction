from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='UserProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('avatar', models.URLField(blank=True)),
                ('role', models.CharField(choices=[('student', 'Student'), ('lecturer', 'Lecturer'), ('admin', 'Admin')], default='student', max_length=20)),
                ('account_code', models.CharField(blank=True, help_text='Student ID or lecturer ID used in analytics views.', max_length=40)),
                ('phone', models.CharField(blank=True, max_length=24)),
                ('gender', models.CharField(blank=True, choices=[('female', 'Female'), ('male', 'Male'), ('other', 'Other')], max_length=16)),
                ('date_of_birth', models.DateField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='profile', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['user__username'],
            },
        ),
    ]
