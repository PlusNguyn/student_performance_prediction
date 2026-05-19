from django.urls import path

from . import views

app_name = 'accounts'

urlpatterns = [
    path('', views.dashboard_redirect, name='home'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('password/', views.change_password_view, name='change_password'),
    path('student/', views.student_dashboard, name='student_dashboard'),
    path('student/<slug:section>/', views.role_page, {'role': 'student'}, name='student_page'),
    path('lecturer/', views.lecturer_dashboard, name='lecturer_dashboard'),
    path('lecturer/<slug:section>/', views.role_page, {'role': 'lecturer'}, name='lecturer_page'),
    path('control/', views.admin_dashboard, name='admin_dashboard'),
    path('control/users/', views.account_list, name='account_list'),
    path('control/users/create/', views.account_create, name='account_create'),
    path('control/users/<int:user_id>/edit/', views.account_update, name='account_update'),
    path('control/users/<int:user_id>/toggle/', views.account_toggle, name='account_toggle'),
    path('control/users/<int:user_id>/reset-password/', views.account_reset_password, name='account_reset_password'),
    path('control/users/<int:user_id>/delete/', views.account_delete, name='account_delete'),
    path('control/<slug:section>/', views.role_page, {'role': 'admin'}, name='admin_page'),
]
