export function TableSkeleton({ rows = 6 }: { rows?: number }) {
  return (
    <div className="skeleton-table" aria-hidden="true">
      {Array.from({ length: rows }).map((_, i) => (
        <div className="skeleton-row" key={i}>
          {Array.from({ length: 7 }).map((__, j) => (
            <span className="skeleton-cell" key={j} style={{ animationDelay: `${(i * 7 + j) * 30}ms` }} />
          ))}
        </div>
      ))}
    </div>
  );
}
