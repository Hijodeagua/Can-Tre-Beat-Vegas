export default function MethodologyPage() {
  return (
    <div className="prose-sm max-w-none">
      <h2 className="text-2xl font-bold">Methodology</h2>

      <Section title="Where the odds come from">
        Odds are pulled from{' '}
        <a
          className="text-blue-600 underline"
          href="https://the-odds-api.com/"
          target="_blank"
          rel="noreferrer"
        >
          The Odds API
        </a>{' '}
        twice daily (a morning snapshot and an evening snapshot before games)
        across major US sportsbooks — DraftKings, FanDuel, BetMGM, BetRivers,
        Bovada, and others. Moneyline (h2h), spread, and totals markets, in
        American odds.
      </Section>

      <Section title="Consensus and no-vig probability">
        The consensus line is the simple average across all books in the
        snapshot. The &ldquo;market&rdquo; win probability is computed by
        converting each side&apos;s average moneyline to an implied
        probability, then normalizing the two sides to sum to 1 — removing the
        bookmaker&apos;s vig.
      </Section>

      <Section title="Line movement">
        Every snapshot is committed to the repo, so each game&apos;s
        &ldquo;opener&rdquo; is the first snapshot in which it appeared. The
        movement chips show how the consensus spread and total have shifted
        since then.
      </Section>

      <Section title="Model picks">
        NBA picks come from a logistic-regression winner model plus a ridge
        score model trained on rolling team stats. The NFL LightGBM models
        (straight-up and against-the-spread) are trained but not yet wired
        into this slate. &ldquo;Edge vs market&rdquo; is the model&apos;s home
        win probability minus the market&apos;s no-vig probability — green or
        red when the disagreement exceeds 5 points.
      </Section>

      <Section title="What this is not">
        A tracker and a public scorecard for my own models — not betting
        advice. Lines move, books differ, and the models are experimental.
      </Section>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-6">
      <h3 className="text-base font-bold">{title}</h3>
      <p className="mt-1 text-sm leading-relaxed text-slate-600">{children}</p>
    </section>
  );
}
