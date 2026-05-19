from django.conf import settings
from django.db import models


class UserProfile(models.Model):
    STUDENT = 'student'
    LECTURER = 'lecturer'
    ADMIN = 'admin'

    ROLE_CHOICES = [
        (STUDENT, 'Student'),
        (LECTURER, 'Lecturer'),
        (ADMIN, 'Admin'),
    ]

    GENDER_CHOICES = [
        ('female', 'Female'),
        ('male', 'Male'),
        ('other', 'Other'),
    ]

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='profile',
    )
    avatar = models.URLField(blank=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=STUDENT)
    account_code = models.CharField(
        max_length=40,
        blank=True,
        help_text='Student ID or lecturer ID used in analytics views.',
    )
    phone = models.CharField(max_length=24, blank=True)
    gender = models.CharField(max_length=16, choices=GENDER_CHOICES, blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['user__username']

    def __str__(self):
        return f'{self.full_name} ({self.get_role_display()})'

    @property
    def full_name(self):
        full_name = self.user.get_full_name()
        return full_name or self.user.username

    @property
    def identifier_label(self):
        if self.role == self.LECTURER:
            return 'Lecturer ID'
        if self.role == self.ADMIN:
            return 'Admin ID'
        return 'Student ID'
