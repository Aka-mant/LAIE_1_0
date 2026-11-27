import uuid
from django.db import models
from django.db.models import Q
from apps.lotteries.models import Lottery, LotteryRule

class Draw(models.Model):
    class Status(models.TextChoices):
        IMPORTED = "IMPORTED", "Импортирован"
        VERIFIED = "VERIFIED", "Проверен"
        QUARANTINED = "QUARANTINED", "Карантин"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    lottery = models.ForeignKey(Lottery, on_delete=models.PROTECT, related_name="draws")
    rule = models.ForeignKey(LotteryRule, on_delete=models.PROTECT, related_name="draws")
    draw_number = models.PositiveBigIntegerField()
    drawn_at = models.DateTimeField()
    source_name = models.CharField(max_length=120)
    source_url = models.URLField(max_length=500, blank=True)
    source_hash = models.CharField(max_length=64)
    jackpot_amount = models.DecimalField(max_digits=18, decimal_places=2, null=True, blank=True)
    total_prize_amount = models.DecimalField(max_digits=18, decimal_places=2, null=True, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.IMPORTED)
    verified_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["lottery", "draw_number"], name="uq_lottery_draw_number")]
        indexes = [models.Index(fields=["lottery", "drawn_at"]), models.Index(fields=["status", "drawn_at"])]
        ordering = ["draw_number"]

    def __str__(self):
        return f"{self.lottery.slug} #{self.draw_number}"

class DrawNumber(models.Model):
    class FieldType(models.TextChoices):
        MAIN = "MAIN", "Основное поле"
        BONUS = "BONUS", "Дополнительное поле"

    draw = models.ForeignKey(Draw, on_delete=models.CASCADE, related_name="numbers")
    value = models.PositiveSmallIntegerField()
    field_type = models.CharField(max_length=8, choices=FieldType.choices)
    position = models.PositiveSmallIntegerField(default=1)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["draw", "value", "field_type"], name="uq_draw_value_field"),
            models.UniqueConstraint(fields=["draw", "position", "field_type"], name="uq_draw_position_field"),
            models.CheckConstraint(condition=Q(value__gte=1), name="ck_draw_number_positive"),
        ]
        ordering = ["field_type", "position"]
