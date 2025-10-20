'use client';

import { useState } from 'react';

export default function ContactPage() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [sent, setSent] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (!name || !email || !message) {
      setError('Please fill in all fields.');
      return;
    }

    try {
      setSending(true);
      const res = await fetch('/api/contact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, message }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data?.error || 'Failed to send');
      setSent(true);
      setName(''); setEmail(''); setMessage('');
    } catch (err: any) {
      setError(err?.message || 'Something went wrong.');
    } finally {
      setSending(false);
    }
  }

  return (
    <section className="container" style={{ padding: '56px 0' }}>
      <h1 style={{ margin: 0, fontSize: 28, lineHeight: 1.15 }}>Contact</h1>
      <p style={{ color: '#666', marginTop: 12 }}>
        Prefer email? <a href="mailto:elassaadlayanne@gmail.com">elassaadlayanne@gmail.com</a> or{' '}
        <a href="mailto:layanne.a@columbia.edu">layanne.a@columbia.edu</a>
      </p>

      <div className="contact-card">
        <form className="contact-form" onSubmit={onSubmit} noValidate>
          <div className="form-row">
            <div className="field">
              <label htmlFor="name">Name</label>
              <input
                id="name" name="name" autoComplete="name"
                value={name} onChange={e => setName(e.target.value)}
                required placeholder="Your name"
                disabled={sending}
              />
            </div>

            <div className="field">
              <label htmlFor="email">Email</label>
              <input
                id="email" name="email" type="email" autoComplete="email"
                value={email} onChange={e => setEmail(e.target.value)}
                required placeholder="you@example.com"
                disabled={sending}
              />
            </div>
          </div>

          <div className="field">
            <label htmlFor="message">Message</label>
            <textarea
              id="message" name="message" rows={6}
              value={message} onChange={e => setMessage(e.target.value)}
              required placeholder="Short note…"
              disabled={sending}
            />
          </div>

          <div className="form-actions">
            <button type="submit" className="cta-solid" disabled={sending}>
              {sending ? 'Sending…' : 'Send message'}
            </button>
            {sent && <span className="form-hint">Thanks — I’ll reply soon.</span>}
            {error && <span className="form-hint" style={{ color: '#b91c1c' }}>{error}</span>}
          </div>
        </form>
      </div>
    </section>
  );
}
