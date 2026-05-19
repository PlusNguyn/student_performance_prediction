from django import forms
from django.contrib.auth.forms import PasswordChangeForm, UserCreationForm
from django.contrib.auth.models import User

from .models import UserProfile


class BootstrapFormMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            if isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs.setdefault('class', 'form-check-input')
            elif isinstance(field.widget, forms.Select):
                field.widget.attrs.setdefault('class', 'form-select')
            else:
                field.widget.attrs.setdefault('class', 'form-control')
            field.widget.attrs.setdefault('placeholder', field.label or field_name.replace('_', ' ').title())


class LoginForm(BootstrapFormMixin, forms.Form):
    username = forms.CharField(label='Username or email')
    password = forms.CharField(widget=forms.PasswordInput, label='Password')
    remember_me = forms.BooleanField(required=False, label='Keep me signed in')


class RegisterForm(BootstrapFormMixin, UserCreationForm):
    email = forms.EmailField(required=True)
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=False)
    role = forms.ChoiceField(
        choices=[
            (UserProfile.STUDENT, 'Student'),
            (UserProfile.LECTURER, 'Lecturer'),
        ]
    )
    account_code = forms.CharField(required=False, label='Student/Lecturer ID')
    phone = forms.CharField(required=False)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = (
            'username',
            'first_name',
            'last_name',
            'email',
            'role',
            'account_code',
            'phone',
            'password1',
            'password2',
        )

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data.get('last_name', '')
        if commit:
            user.save()
            UserProfile.objects.update_or_create(
                user=user,
                defaults={
                    'role': self.cleaned_data['role'],
                    'account_code': self.cleaned_data.get('account_code', ''),
                    'phone': self.cleaned_data.get('phone', ''),
                },
            )
        return user


class AccountCreateForm(RegisterForm):
    role = forms.ChoiceField(choices=UserProfile.ROLE_CHOICES)


class AccountUpdateForm(BootstrapFormMixin, forms.Form):
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=False)
    email = forms.EmailField(required=True)
    role = forms.ChoiceField(choices=UserProfile.ROLE_CHOICES)
    account_code = forms.CharField(required=False, label='Student/Lecturer ID')
    phone = forms.CharField(required=False)
    gender = forms.ChoiceField(required=False, choices=[('', 'Not specified')] + UserProfile.GENDER_CHOICES)
    date_of_birth = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    avatar = forms.URLField(required=False)
    is_active = forms.BooleanField(required=False, label='Account enabled')

    def __init__(self, *args, **kwargs):
        self.user_instance = kwargs.pop('user_instance')
        profile, _ = UserProfile.objects.get_or_create(user=self.user_instance)
        initial = {
            'first_name': self.user_instance.first_name,
            'last_name': self.user_instance.last_name,
            'email': self.user_instance.email,
            'role': profile.role,
            'account_code': profile.account_code,
            'phone': profile.phone,
            'gender': profile.gender,
            'date_of_birth': profile.date_of_birth,
            'avatar': profile.avatar,
            'is_active': self.user_instance.is_active,
        }
        kwargs.setdefault('initial', initial)
        super().__init__(*args, **kwargs)

    def save(self):
        user = self.user_instance
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data.get('last_name', '')
        user.email = self.cleaned_data['email']
        user.is_active = self.cleaned_data.get('is_active', False)
        user.save(update_fields=['first_name', 'last_name', 'email', 'is_active'])
        UserProfile.objects.update_or_create(
            user=user,
            defaults={
                'role': self.cleaned_data['role'],
                'account_code': self.cleaned_data.get('account_code', ''),
                'phone': self.cleaned_data.get('phone', ''),
                'gender': self.cleaned_data.get('gender', ''),
                'date_of_birth': self.cleaned_data.get('date_of_birth'),
                'avatar': self.cleaned_data.get('avatar', ''),
            },
        )
        return user


class ProfileUpdateForm(AccountUpdateForm):
    role = forms.ChoiceField(choices=UserProfile.ROLE_CHOICES, disabled=True)
    is_active = forms.BooleanField(required=False, disabled=True, label='Account enabled')


class AdminPasswordResetForm(BootstrapFormMixin, forms.Form):
    new_password = forms.CharField(widget=forms.PasswordInput, min_length=8)


class StyledPasswordChangeForm(BootstrapFormMixin, PasswordChangeForm):
    pass
