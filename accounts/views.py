import json
from functools import wraps

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Q
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from .forms import (
    AccountCreateForm,
    AccountUpdateForm,
    AdminPasswordResetForm,
    LoginForm,
    ProfileUpdateForm,
    RegisterForm,
    StyledPasswordChangeForm,
)
from .models import UserProfile


def chart_config(config):
    return json.dumps(config, separators=(',', ':'))


def get_profile(user):
    if not user.is_authenticated:
        return None
    default_role = UserProfile.ADMIN if user.is_superuser else UserProfile.STUDENT
    profile, _ = UserProfile.objects.get_or_create(user=user, defaults={'role': default_role})
    return profile


def role_for_user(user):
    if not user.is_authenticated:
        return 'guest'
    if user.is_superuser:
        get_profile(user)
        return UserProfile.ADMIN
    profile = get_profile(user)
    if profile and profile.role:
        return profile.role
    if user.is_staff:
        return UserProfile.LECTURER
    return UserProfile.STUDENT


def dashboard_url_for_role(role):
    urls = {
        UserProfile.ADMIN: 'accounts:admin_dashboard',
        UserProfile.LECTURER: 'accounts:lecturer_dashboard',
        UserProfile.STUDENT: 'accounts:student_dashboard',
    }
    return reverse(urls.get(role, 'accounts:student_dashboard'))


def role_required(*allowed_roles):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect(f"{reverse('accounts:login')}?next={request.path}")
            role = role_for_user(request.user)
            if role == UserProfile.ADMIN or role in allowed_roles:
                return view_func(request, *args, **kwargs)
            messages.error(request, 'You do not have permission to access that workspace.')
            return redirect(dashboard_url_for_role(role))

        return wrapper

    return decorator


def dashboard_redirect(request):
    if not request.user.is_authenticated:
        return redirect('accounts:login')
    return redirect(dashboard_url_for_role(role_for_user(request.user)))


def login_view(request):
    if request.user.is_authenticated:
        return redirect(dashboard_url_for_role(role_for_user(request.user)))

    form = LoginForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        username = form.cleaned_data['username'].strip()
        password = form.cleaned_data['password']
        lookup_username = username
        if '@' in username:
            matched_user = User.objects.filter(email__iexact=username).first()
            if matched_user:
                lookup_username = matched_user.username

        user = authenticate(request, username=lookup_username, password=password)
        if user is not None:
            if not user.is_active:
                form.add_error(None, 'This account is disabled.')
            else:
                login(request, user)
                if not form.cleaned_data.get('remember_me'):
                    request.session.set_expiry(0)
                messages.success(request, 'Welcome back. Your dashboard is ready.')
                return redirect(request.GET.get('next') or dashboard_url_for_role(role_for_user(user)))
        else:
            form.add_error(None, 'Invalid username/email or password.')

    return render(request, 'auth/login.html', {'form': form, 'page_title': 'Sign in'})


def register_view(request):
    if request.user.is_authenticated:
        return redirect(dashboard_url_for_role(role_for_user(request.user)))

    form = RegisterForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, 'Your account has been created.')
        return redirect(dashboard_url_for_role(role_for_user(user)))
    return render(request, 'auth/register.html', {'form': form, 'page_title': 'Create account'})


def logout_view(request):
    logout(request)
    messages.info(request, 'You have signed out.')
    return redirect('accounts:login')


@login_required
def profile_view(request):
    form = ProfileUpdateForm(request.POST or None, user_instance=request.user)
    if request.method == 'POST' and form.is_valid():
        form.save()
        messages.success(request, 'Profile updated.')
        return redirect('accounts:profile')
    return render(
        request,
        'auth/profile.html',
        {
            'form': form,
            'page_title': 'Profile',
            'active_role': role_for_user(request.user),
            'active_section': 'profile',
        },
    )


@login_required
def change_password_view(request):
    form = StyledPasswordChangeForm(request.user, request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        update_session_auth_hash(request, user)
        messages.success(request, 'Password changed successfully.')
        return redirect('accounts:profile')
    return render(
        request,
        'auth/password_change.html',
        {
            'form': form,
            'page_title': 'Change password',
            'active_role': role_for_user(request.user),
            'active_section': 'settings',
        },
    )


def student_dashboard_data(student_id):
    return {
        'student_id': student_id,
        'kpis': [
            {'label': 'Knowledge absorption', 'value': '82%', 'trend': '+6.4%', 'icon': 'fa-brain', 'tone': 'primary'},
            {'label': 'Learning score', 'value': '7.8', 'trend': '+0.5', 'icon': 'fa-chart-line', 'tone': 'success'},
            {'label': 'Risk level', 'value': 'Medium', 'trend': 'Watch', 'icon': 'fa-triangle-exclamation', 'tone': 'warning'},
            {'label': 'Engagement score', 'value': '74', 'trend': '-3 pts', 'icon': 'fa-bolt', 'tone': 'info'},
        ],
        'status': {
            'label': 'Predicted learning status',
            'value': 'On track with intervention',
            'confidence': '88%',
            'description': 'Assignment timing and quiz variance are the strongest current signals.',
        },
        'alerts': [
            {'title': 'Interaction is decreasing', 'body': 'Your activity dropped across the latest two sessions.', 'tone': 'warning'},
            {'title': 'Finish assignments earlier', 'body': 'Submitting 24 hours earlier can improve the forecasted score.', 'tone': 'info'},
            {'title': 'Quiz recovery window', 'body': 'Two focused quiz reviews this week are recommended.', 'tone': 'success'},
        ],
        'recommendations': [
            {'title': 'Review weak topics', 'body': 'Spend 30 minutes on unit 4 videos before the next quiz.', 'impact': 'High impact'},
            {'title': 'Stabilize schedule', 'body': 'Keep at least four active learning days this week.', 'impact': 'Medium impact'},
            {'title': 'Ask for feedback', 'body': 'Request lecturer comments on the last assignment attempt.', 'impact': 'Medium impact'},
        ],
        'history': [
            {'date': 'May 13', 'event': 'Prediction updated', 'detail': 'Risk changed from High to Medium'},
            {'date': 'May 15', 'event': 'Quiz completed', 'detail': 'Score 78%, +9 points from previous quiz'},
            {'date': 'May 18', 'event': 'Assignment submitted', 'detail': 'Submitted 8 hours before deadline'},
        ],
        'feature_rows': [
            {'feature': 'Quiz average', 'weight': '28%', 'direction': 'Positive'},
            {'feature': 'Late submissions', 'weight': '22%', 'direction': 'Negative'},
            {'feature': 'Weekly activity', 'weight': '19%', 'direction': 'Positive'},
            {'feature': 'Forum engagement', 'weight': '13%', 'direction': 'Neutral'},
        ],
        'heatmap': [3, 5, 4, 1, 6, 7, 2, 4, 6, 3, 2, 7, 5, 4, 3, 6, 4, 2, 1, 5, 7, 6, 3, 4, 5, 6, 2, 3],
        'charts': {
            'learning': chart_config({
                'type': 'line',
                'data': {
                    'labels': ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8'],
                    'datasets': [
                        {'label': 'Learning score', 'data': [62, 66, 64, 70, 74, 73, 79, 82], 'borderColor': '#2563eb', 'backgroundColor': 'rgba(37,99,235,.16)', 'fill': True, 'tension': .42},
                        {'label': 'Engagement', 'data': [78, 76, 72, 75, 79, 77, 75, 74], 'borderColor': '#14b8a6', 'backgroundColor': 'rgba(20,184,166,.12)', 'fill': True, 'tension': .42},
                    ],
                },
                'options': {'plugins': {'legend': {'display': True}}, 'scales': {'y': {'min': 40, 'max': 100}}},
            }),
            'quiz': chart_config({
                'type': 'bar',
                'data': {
                    'labels': ['Quiz 1', 'Quiz 2', 'Quiz 3', 'Quiz 4', 'Quiz 5'],
                    'datasets': [{'label': 'Score', 'data': [68, 74, 71, 79, 83], 'backgroundColor': ['#60a5fa', '#2dd4bf', '#f59e0b', '#818cf8', '#34d399']}],
                },
                'options': {'plugins': {'legend': {'display': False}}, 'scales': {'y': {'min': 0, 'max': 100}}},
            }),
            'assignment': chart_config({
                'type': 'doughnut',
                'data': {
                    'labels': ['Completed', 'Pending', 'Late'],
                    'datasets': [{'data': [76, 14, 10], 'backgroundColor': ['#22c55e', '#f59e0b', '#ef4444'], 'borderWidth': 0}],
                },
                'options': {'cutout': '72%', 'plugins': {'legend': {'position': 'bottom'}}},
            }),
        },
    }


@role_required(UserProfile.STUDENT)
def student_dashboard(request):
    profile = get_profile(request.user)
    student_id = request.GET.get('student_id') or (profile.account_code if profile and profile.role == UserProfile.STUDENT else '')
    context = {
        'page_title': 'Student Dashboard',
        'active_role': UserProfile.STUDENT,
        'active_section': 'dashboard',
        'student_id': student_id,
    }
    if student_id:
        context['dashboard'] = student_dashboard_data(student_id)
    else:
        context['empty_message'] = 'Enter a student ID to view predictions and analytics.'
    return render(request, 'student/dashboard.html', context)


@role_required(UserProfile.LECTURER)
def lecturer_dashboard(request):
    context = {
        'page_title': 'Lecturer Dashboard',
        'active_role': UserProfile.LECTURER,
        'active_section': 'dashboard',
        'kpis': [
            {'label': 'Total students', 'value': '248', 'trend': '+12 this term', 'icon': 'fa-users', 'tone': 'primary'},
            {'label': 'High risk', 'value': '31', 'trend': '-8%', 'icon': 'fa-triangle-exclamation', 'tone': 'danger'},
            {'label': 'Class absorption', 'value': '78%', 'trend': '+4%', 'icon': 'fa-graduation-cap', 'tone': 'success'},
            {'label': 'Course completion', 'value': '69%', 'trend': '+7%', 'icon': 'fa-list-check', 'tone': 'info'},
        ],
        'weak_students': [
            {'name': 'Nguyen Minh Anh', 'student_id': 'SV1024', 'risk': 'High', 'score': 51, 'engagement': 42},
            {'name': 'Tran Quoc Bao', 'student_id': 'SV1108', 'risk': 'High', 'score': 48, 'engagement': 39},
            {'name': 'Le Thanh Ha', 'student_id': 'SV1201', 'risk': 'Medium', 'score': 58, 'engagement': 55},
            {'name': 'Pham Gia Huy', 'student_id': 'SV1302', 'risk': 'Medium', 'score': 62, 'engagement': 57},
        ],
        'alerts': [
            {'title': '31 students need attention', 'body': 'Prioritize students with low attendance and declining quiz trend.', 'tone': 'danger'},
            {'title': 'Assignment delay spike', 'body': 'Late submissions increased in module 5.', 'tone': 'warning'},
            {'title': 'Model summary', 'body': 'Production model accuracy is stable at 91.8%.', 'tone': 'success'},
        ],
        'charts': {
            'engagement': chart_config({
                'type': 'bar',
                'data': {'labels': ['0-20', '21-40', '41-60', '61-80', '81-100'], 'datasets': [{'label': 'Students', 'data': [8, 26, 61, 97, 56], 'backgroundColor': '#2563eb'}]},
                'options': {'plugins': {'legend': {'display': False}}},
            }),
            'activity': chart_config({
                'type': 'line',
                'data': {'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 'datasets': [{'label': 'Learning sessions', 'data': [420, 510, 498, 620, 588, 361, 292], 'borderColor': '#14b8a6', 'backgroundColor': 'rgba(20,184,166,.14)', 'fill': True, 'tension': .4}]},
                'options': {'plugins': {'legend': {'display': False}}},
            }),
            'risk': chart_config({
                'type': 'pie',
                'data': {'labels': ['Low', 'Medium', 'High'], 'datasets': [{'data': [139, 78, 31], 'backgroundColor': ['#22c55e', '#f59e0b', '#ef4444'], 'borderWidth': 0}]},
                'options': {'plugins': {'legend': {'position': 'bottom'}}},
            }),
            'model': chart_config({
                'type': 'radar',
                'data': {'labels': ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'], 'datasets': [{'label': 'Production', 'data': [92, 89, 87, 88, 94], 'borderColor': '#6366f1', 'backgroundColor': 'rgba(99,102,241,.18)'}]},
                'options': {'scales': {'r': {'min': 70, 'max': 100}}},
            }),
        },
    }
    return render(request, 'lecturer/dashboard.html', context)


@role_required(UserProfile.ADMIN)
def admin_dashboard(request):
    try:
        total_users = User.objects.count()
        active_users = User.objects.filter(is_active=True).count()
        total_students = UserProfile.objects.filter(role=UserProfile.STUDENT).count()
        total_lecturers = UserProfile.objects.filter(role=UserProfile.LECTURER).count()
    except Exception:
        total_users = active_users = total_students = total_lecturers = 0

    context = {
        'page_title': 'Admin Dashboard',
        'active_role': UserProfile.ADMIN,
        'active_section': 'dashboard',
        'kpis': [
            {'label': 'Total users', 'value': total_users, 'trend': f'{active_users} active', 'icon': 'fa-user-shield', 'tone': 'primary'},
            {'label': 'Students', 'value': total_students, 'trend': 'learning profiles', 'icon': 'fa-user-graduate', 'tone': 'success'},
            {'label': 'Lecturers', 'value': total_lecturers, 'trend': 'teaching accounts', 'icon': 'fa-chalkboard-user', 'tone': 'info'},
            {'label': 'Model accuracy', 'value': '91.8%', 'trend': '+1.2%', 'icon': 'fa-microchip', 'tone': 'warning'},
        ],
        'services': [
            {'name': 'Airflow DAG', 'status': 'Healthy', 'detail': 'student_performance_weekly_retraining_pipeline', 'tone': 'success'},
            {'name': 'MLflow registry', 'status': 'Synced', 'detail': 'production alias resolved', 'tone': 'success'},
            {'name': 'Prediction API', 'status': 'Normal', 'detail': '12.4k requests this week', 'tone': 'info'},
            {'name': 'Model retraining', 'status': 'Scheduled', 'detail': 'Next run Monday 02:00', 'tone': 'warning'},
        ],
        'logs': [
            {'time': '09:12', 'source': 'auth', 'message': 'Lecturer account enabled'},
            {'time': '10:04', 'source': 'prediction', 'message': 'Batch prediction completed for class DS101'},
            {'time': '11:31', 'source': 'mlflow', 'message': 'Model version 7 promoted to staging'},
            {'time': '14:22', 'source': 'airflow', 'message': 'Feature pipeline finished successfully'},
        ],
        'charts': {
            'prediction': chart_config({
                'type': 'line',
                'data': {'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], 'datasets': [{'label': 'Predictions', 'data': [4200, 5100, 6300, 7200, 8100, 9300], 'borderColor': '#2563eb', 'backgroundColor': 'rgba(37,99,235,.15)', 'fill': True, 'tension': .4}]},
                'options': {'plugins': {'legend': {'display': False}}},
            }),
            'requests': chart_config({
                'type': 'bar',
                'data': {'labels': ['API', 'Batch', 'Dashboard', 'Exports'], 'datasets': [{'label': 'Requests', 'data': [12400, 3100, 8800, 740], 'backgroundColor': ['#2563eb', '#14b8a6', '#f59e0b', '#6366f1']}]},
                'options': {'plugins': {'legend': {'display': False}}},
            }),
            'model': chart_config({
                'type': 'bar',
                'data': {'labels': ['Accuracy', 'Precision', 'Recall', 'F1'], 'datasets': [{'label': 'Metric', 'data': [91.8, 89.6, 87.2, 88.4], 'backgroundColor': '#14b8a6'}]},
                'options': {'indexAxis': 'y', 'plugins': {'legend': {'display': False}}, 'scales': {'x': {'min': 70, 'max': 100}}},
            }),
        },
    }
    return render(request, 'admin/dashboard.html', context)


SECTION_TITLES = {
    'progress': 'Learning Progress',
    'predictions': 'Predictions',
    'analytics': 'Analytics',
    'notifications': 'Notifications',
    'recommendations': 'Recommendations',
    'settings': 'Settings',
    'students': 'Students',
    'classes': 'Classes',
    'reports': 'Reports',
    'alerts': 'Alerts',
    'models': 'Model Management',
    'mlflow': 'MLflow Monitoring',
    'airflow': 'Airflow Monitoring',
    'logs': 'System Logs',
    'monitoring': 'System Monitoring',
}


@login_required
def role_page(request, role, section):
    guard = role_required(role)
    if role != UserProfile.ADMIN:
        return guard(_role_page)(request, role, section)
    return role_required(UserProfile.ADMIN)(_role_page)(request, role, section)


def _role_page(request, role, section):
    title = SECTION_TITLES.get(section, section.replace('-', ' ').title())
    return render(
        request,
        'shared/role_page.html',
        {
            'page_title': title,
            'active_role': role,
            'active_section': section,
            'panels': build_section_panels(role, section),
            'section_chart': chart_config({
                'type': 'line',
                'data': {
                    'labels': ['W1', 'W2', 'W3', 'W4', 'W5', 'W6'],
                    'datasets': [{'label': title, 'data': [58, 64, 71, 69, 77, 83], 'borderColor': '#2563eb', 'backgroundColor': 'rgba(37,99,235,.14)', 'fill': True, 'tension': .4}],
                },
                'options': {'plugins': {'legend': {'display': False}}},
            }),
        },
    )


def build_section_panels(role, section):
    base = [
        {'title': 'Overview', 'body': f'{SECTION_TITLES.get(section, section.title())} summary is ready for this workspace.', 'icon': 'fa-layer-group'},
        {'title': 'Recent activity', 'body': 'Latest updates, alerts, and actions appear here.', 'icon': 'fa-clock-rotate-left'},
        {'title': 'Export-ready data', 'body': 'Tables are structured for future PDF/Excel export integration.', 'icon': 'fa-file-export'},
    ]
    if role == UserProfile.STUDENT:
        base.append({'title': 'AI recommendation', 'body': 'Personalized interventions and feature explanations are grouped for review.', 'icon': 'fa-wand-magic-sparkles'})
    elif role == UserProfile.LECTURER:
        base.append({'title': 'Class filter', 'body': 'Risk level, score, and attendance filters can be wired to live class data.', 'icon': 'fa-filter'})
    else:
        base.append({'title': 'Operations control', 'body': 'MLOps services, logs, model registry, and account workflows are grouped by priority.', 'icon': 'fa-server'})
    return base


@role_required(UserProfile.ADMIN)
def account_list(request):
    query = request.GET.get('q', '').strip()
    role = request.GET.get('role', '').strip()
    users = User.objects.all().order_by('first_name', 'username')
    if query:
        users = users.filter(
            Q(username__icontains=query)
            | Q(first_name__icontains=query)
            | Q(last_name__icontains=query)
            | Q(email__icontains=query)
            | Q(profile__account_code__icontains=query)
        )
    if role:
        users = users.filter(profile__role=role)

    rows = []
    for user in users[:250]:
        profile = get_profile(user)
        rows.append({'user': user, 'profile': profile, 'role': role_for_user(user)})

    return render(
        request,
        'admin/users_list.html',
        {
            'page_title': 'User Management',
            'active_role': UserProfile.ADMIN,
            'active_section': 'users',
            'rows': rows,
            'query': query,
            'role_filter': role,
            'role_choices': UserProfile.ROLE_CHOICES,
        },
    )


@role_required(UserProfile.ADMIN)
def account_create(request):
    form = AccountCreateForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        messages.success(request, f'Account {user.username} created.')
        return redirect('accounts:account_list')
    return render(
        request,
        'admin/user_form.html',
        {
            'form': form,
            'page_title': 'Create Account',
            'active_role': UserProfile.ADMIN,
            'active_section': 'users',
            'submit_label': 'Create account',
        },
    )


@role_required(UserProfile.ADMIN)
def account_update(request, user_id):
    target = get_object_or_404(User, pk=user_id)
    form = AccountUpdateForm(request.POST or None, user_instance=target)
    if request.method == 'POST' and form.is_valid():
        form.save()
        messages.success(request, f'Account {target.username} updated.')
        return redirect('accounts:account_list')
    return render(
        request,
        'admin/user_form.html',
        {
            'form': form,
            'page_title': 'Update Account',
            'target_user': target,
            'active_role': UserProfile.ADMIN,
            'active_section': 'users',
            'submit_label': 'Save changes',
        },
    )


@role_required(UserProfile.ADMIN)
def account_toggle(request, user_id):
    target = get_object_or_404(User, pk=user_id)
    if target == request.user:
        messages.error(request, 'You cannot disable your own active session.')
    else:
        target.is_active = not target.is_active
        target.save(update_fields=['is_active'])
        messages.success(request, f'{target.username} is now {"enabled" if target.is_active else "disabled"}.')
    return redirect('accounts:account_list')


@role_required(UserProfile.ADMIN)
def account_reset_password(request, user_id):
    target = get_object_or_404(User, pk=user_id)
    form = AdminPasswordResetForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        target.set_password(form.cleaned_data['new_password'])
        target.save(update_fields=['password'])
        messages.success(request, f'Password reset for {target.username}.')
        return redirect('accounts:account_list')
    return render(
        request,
        'admin/reset_password.html',
        {
            'form': form,
            'page_title': 'Reset Password',
            'target_user': target,
            'active_role': UserProfile.ADMIN,
            'active_section': 'users',
        },
    )


@role_required(UserProfile.ADMIN)
def account_delete(request, user_id):
    target = get_object_or_404(User, pk=user_id)
    if target == request.user:
        messages.error(request, 'You cannot delete your own account.')
        return redirect('accounts:account_list')
    if request.method == 'POST':
        username = target.username
        target.delete()
        messages.success(request, f'Account {username} deleted.')
        return redirect('accounts:account_list')
    return render(
        request,
        'admin/confirm_delete.html',
        {
            'page_title': 'Delete Account',
            'target_user': target,
            'active_role': UserProfile.ADMIN,
            'active_section': 'users',
        },
    )
