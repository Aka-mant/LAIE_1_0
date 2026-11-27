import uuid
from django.core.validators import MinValueValidator
from django.db import models

class Lottery(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=160)
    slug = models.SlugField(max_length=80, unique=True)
    country_code = models.CharField(max_length=2, default="RU")
    operator_name = models.CharField(max_length=160, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "Лотерея"
        verbose_name_plural = "Лотереи"

    def __str__(self):
        return self.name

class LotteryRule(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    lottery = models.ForeignKey(Lottery, on_delete=models.PROTECT, related_name="rules")
    version = models.CharField(max_length=40)
    main_pool_size = models.PositiveSmallIntegerField()
    main_numbers_count = models.PositiveSmallIntegerField()
    bonus_pool_size = models.PositiveSmallIntegerField(default=0)
    bonus_numbers_count = models.PositiveSmallIntegerField(default=0)
    base_price = models.DecimalField(max_digits=10, decimal_places=2, validators=[MinValueValidator(0)])
    currency = models.CharField(max_length=3, default="RUB")
    prize_fund_share = models.DecimalField(max_digits=5, decimal_places=4, null=True, blank=True)
    effective_from = models.DateTimeField()
    effective_until = models.DateTimeField(null=True, blank=True)
    official_source_url = models.URLField(max_length=500, blank=True)
    source_snapshot_hash = models.CharField(max_length=64, blank=True)
    is_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["lottery", "version"], name="uq_lottery_rule_version"),
            models.CheckConstraint(condition=models.Q(main_pool_size__gte=models.F("main_numbers_count")), name="ck_main_pool_ge_count"),
            models.CheckConstraint(condition=models.Q(bonus_pool_size__gte=models.F("bonus_numbers_count")), name="ck_bonus_pool_ge_count"),
            models.CheckConstraint(condition=models.Q(effective_until__isnull=True) | models.Q(effective_until__gt=models.F("effective_from")), name="ck_rule_dates_order"),
        ]
        ordering = ["-effective_from"]

    def __str__(self):
        return f"{self.lottery} / {self.version}"
