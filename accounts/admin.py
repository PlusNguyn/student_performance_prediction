from django.contrib import admin

from .models import UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = (
        'full_name',
        'role',
        'account_code',
        'phone',
        'created_at',
    )
    list_filter = ('role', 'gender', 'created_at')
    search_fields = (
        'user__username',
        'user__first_name',
        'user__last_name',
        'user__email',
        'account_code',
        'phone',
    )
    autocomplete_fields = ('user',)
