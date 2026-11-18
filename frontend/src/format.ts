const integer = new Intl.NumberFormat("en-US");

export function count(value: number | null | undefined): string {
  return value == null ? "—" : integer.format(value);
}

export function metric(value: number | null | undefined): string {
  return value == null ? "—" : value.toFixed(4);
}

export function ms(value: number | null | undefined): string {
  return value == null ? "—" : `${value.toFixed(1)} ms`;
}

export function shortHash(value: string | null | undefined): string {
  return value ? `${value.slice(0, 8)}…` : "—";
}

export function pixel(value: number | null | undefined): string {
  return value == null ? "—" : value.toFixed(3);
}
