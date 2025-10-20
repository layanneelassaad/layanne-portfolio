import { NextResponse } from 'next/server';
import nodemailer from 'nodemailer';
import { z } from 'zod';
export const runtime = 'nodejs';


const schema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email().max(200),
  message: z.string().min(2).max(5000),
});

export async function POST(req: Request) {
  try {
    const json = await req.json();
    const { name, email, message } = schema.parse(json);

    // env: SMTP_* + EMAIL_FROM + EMAIL_TO (comma-separated allowed)
    const {
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS,
      EMAIL_FROM, EMAIL_TO,
    } = process.env;

    if (!SMTP_HOST || !SMTP_PORT || !SMTP_USER || !SMTP_PASS || !EMAIL_FROM || !EMAIL_TO) {
      return NextResponse.json({ ok: false, error: 'Server email not configured.' }, { status: 500 });
    }

    const transporter = nodemailer.createTransport({
      host: SMTP_HOST,
      port: Number(SMTP_PORT),
      secure: Number(SMTP_PORT) === 465,
      auth: { user: SMTP_USER, pass: SMTP_PASS },
    });

    await transporter.sendMail({
      from: EMAIL_FROM,
      to: EMAIL_TO, // e.g. "elassaadlayanne@gmail.com, layanne.a@columbia.edu"
      subject: `Portfolio inquiry — ${name}`,
      replyTo: email,
      text: `${message}\n\n— ${name}\n${email}`,
      html: `
        <div style="font-family:system-ui,Segoe UI,Roboto">
          <p>${message.replace(/\n/g,'<br/>')}</p>
          <p style="color:#6b7280">— ${name}<br/>${email}</p>
        </div>
      `,
    });

    return NextResponse.json({ ok: true });
  } catch (err: any) {
    const msg = err?.issues?.[0]?.message || err?.message || 'Unknown error';
    return NextResponse.json({ ok: false, error: msg }, { status: 400 });
  }
}
