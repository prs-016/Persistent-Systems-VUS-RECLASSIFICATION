interface Props {
  theme: "light" | "dark";
  onToggleTheme: () => void;
}

export function Header({ theme, onToggleTheme }: Props) {
  return (
    <header className="site-header">
      <div className="header-inner">
        <div className="wordmark">
          <span className="wordmark-main">VUS Reclassification</span>
          <span className="wordmark-sub">Persistent Systems</span>
        </div>
        <nav className="header-nav">
          <a href="#upload">Upload</a>
          <button
            type="button"
            className="theme-toggle"
            onClick={onToggleTheme}
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            <span className="dot" />
            {theme === "dark" ? "Dark" : "Light"}
          </button>
        </nav>
      </div>
    </header>
  );
}
